
import torch
import os
import sys
from PIL import Image
from typing import Dict

# 确保路径正确
sys.path.append("/data01/kjt_data/MLLM/DQLLaVAhpc")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("base_model.model."):
            new_key = new_key[len("base_model.model."):]
        elif new_key.startswith("model."):
            new_key = new_key[len("model."):]
        normalized[new_key] = value
    return normalized


def _find_stage1_weight_file(path: str):
    if os.path.isfile(path) and path.endswith(".bin"):
        return path

    if not os.path.isdir(path):
        return None

    candidates = [
        os.path.join(path, "stage_trainables.bin"),
        os.path.join(path, "non_lora_trainables.bin"),
        os.path.join(path, "mm_projector.bin"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


class Llava:
    def __init__(self, model_path, args):
        print(f"Loading DQ-LLaVA model from: {model_path}")
        model_base = os.environ.get(
            "DQ_BASE_MODEL_PATH",
            "/data01/kjt_data/MLLM/LLaVARadWeight/vicuna-7b-v1.5",
        )
        arch_model_path = os.environ.get(
            "DQ_ARCH_MODEL_PATH",
            "/data01/kjt_data/MLLM/LLaVARadWeight",
        )
        stage1_weight_path = _find_stage1_weight_file(model_path)
        
        disable_torch_init()

        try:
            if stage1_weight_path is not None:
                resolved_arch_path = arch_model_path if os.path.exists(arch_model_path) else model_path
                print(f"Detected stage1 weights: {stage1_weight_path}")
                print(f"Loading architecture/config from: {resolved_arch_path}")
                print(f"Loading base model from: {model_base}")

                # 先按 base + llava 配置加载，再叠加 stage1 训练得到的权重
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    model_path=resolved_arch_path,
                    model_base=model_base,
                    model_name="llavarad",
                    load_8bit=False,
                    load_4bit=False,
                    device_map="auto",
                )

                stage1_state_dict = torch.load(stage1_weight_path, map_location="cpu")
                stage1_state_dict = _normalize_state_dict_keys(stage1_state_dict)
                load_msg = self.model.load_state_dict(stage1_state_dict, strict=False)
                print(
                    f"Stage1 weights loaded with strict=False "
                    f"(missing_keys={len(load_msg.missing_keys)}, unexpected_keys={len(load_msg.unexpected_keys)})"
                )
            else:
                # 默认兼容：当 model_path 本身就是完整模型目录时按原逻辑直接加载
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name="llavarad",
                    load_8bit=False,
                    load_4bit=False,
                    device_map="auto",
                )

            self.model.eval() # 确保进入 eval 模式
            print("Model loaded successfully!")

            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        self.conv_mode = "v1"

    def process_messages(self, messages):
        # 1. 构建 Prompt
        system_msg = messages.get("system", "")
        user_msg = messages.get("prompt", "")
        
        # 确保 prompt 中有 <image> 占位符，否则 prepare_inputs_labels_for_multimodal 可能不会处理图像
        if DEFAULT_IMAGE_TOKEN not in user_msg:
            user_msg = DEFAULT_IMAGE_TOKEN + "\n" + user_msg
            
        full_prompt = f"{system_msg}\n{user_msg}" if system_msg else user_msg

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # 2. 处理图像
        image_tensor = None
        if "images" in messages:
            images = messages["images"]
            if not isinstance(images, list): images = [images]
            images = [img if isinstance(img, Image.Image) else Image.open(img).convert('RGB') for img in images]
            
            # 使用模型对应的 image_processor
            image_tensor = process_images(images, self.image_processor, self.model.config)
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # 3. 转化为 Input IDs
        input_ids = tokenizer_image_token(
            prompt_text, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)

        return {
            'input_ids': input_ids,
            'images': image_tensor
        }

    def generate_output(self, messages):
        inputs = self.process_messages(messages)

        
        # 这里的 images 会传给 forward -> prepare_inputs_labels_for_multimodal
        # 从而在你的模型内部生成 V_pool
        generation_kwargs = {
            "input_ids": inputs['input_ids'],
            "images": inputs['images'],
            "do_sample": True if self.temperature > 0 else False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
        }

        with torch.inference_mode():
            output_ids = self.model.generate(**generation_kwargs)

        # 解码
        input_length = inputs['input_ids'].shape[1]
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, input_length:], 
            skip_special_tokens=True
        )[0]
        print("Generated_text:")
        print(generated_text)
        
        return generated_text.strip()
    
    def generate_outputs(self, messages_list):
        results = []
        for messages in messages_list:
            result = self.generate_output(messages)
            results.append(result)
        return results


# import os
# import torch
# non_lora_trainables = torch.load('/data_lyn/DQ_LLaVA/checkpoints/b1visfonly11-lora-3e-1e-4/adapter_model.bin', map_location='cpu')
# non_lora_trainables