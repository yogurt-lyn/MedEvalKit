import torch
import os
import sys
import json
from PIL import Image
# sys.path.append("/data_lyn/DQ_LLaVA")
sys.path.append("/data_lyn/LLaVARad")

# export CUDA_HOME=/usr/local/cuda-11.8
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda-11.8/bin:$PATH

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images

class Llava:
    def __init__(self, model_path, args):
        super().__init__()
        
        print(f"Loading LLaVA-Rad model from: {model_path}")
        model_base = os.path.join("/data_lyn/LLaVARadWeight", "vicuna-7b-v1.5")
        

        disable_torch_init()

        try:

            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=model_base,
                model_name="llavarad",
                load_8bit=False,
                load_4bit=False,
                device_map="auto"
            )
            print("LLaVA-Rad model loaded successfully!")
            
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
    
    # def process_messages(self, messages):

    #     prompt = ""
    #     images = None
        

    #     if "system" in messages:
    #         prompt = messages["system"]
        

    #     if "prompt" in messages:
    #         if prompt:
    #             prompt = prompt + "\n" + messages["prompt"]
    #         else:
    #             prompt = messages["prompt"]
        

    #     if "images" in messages:
    #         images = messages["images"]
    #         if not isinstance(images, list):
    #             images = [images]
    #         images = [img if isinstance(img, Image.Image) else Image.open(img).convert('RGB') for img in images]
        
    #     else:
    #         images = None
        
    #     conv = conv_templates[self.conv_mode].copy()



    #     if images:

    #         if len(images) == 1:
    #             image_tokens = DEFAULT_IMAGE_TOKEN
    #         else:
    #             image_tokens = "\n".join([DEFAULT_IMAGE_TOKEN] * len(images))
            
    #         if prompt:
    #             is_token_present=prompt.strip().startswith(DEFAULT_IMAGE_TOKEN)

    #             if is_token_present:
    #                 query = prompt
    #             else:
    #                 query = query = f"{image_tokens}\n{prompt}"

    #         else:
    #             query = image_tokens
    #     else:
    #         query = prompt if prompt else ""

    #     query = prompt

        
    #     conv.append_message(conv.roles[0], query)
    #     conv.append_message(conv.roles[1], None)
    #     prompt_text = conv.get_prompt()
        
    #     if images:

    #         image_tensor = process_images(images, self.image_processor, self.model.config)
    #         if type(image_tensor) is list:
    #             image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
    #         else:
    #             image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
    #     else:
    #         image_tensor = None
        
    #     input_ids = tokenizer_image_token(
    #         prompt_text, 
    #         self.tokenizer, 
    #         IMAGE_TOKEN_INDEX, 
    #         return_tensors='pt'
    #     ).unsqueeze(0).to(self.model.device)
        
    #     return {
    #         'input_ids': input_ids,
    #         'images': image_tensor
    #     }
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
        """生成输出"""
        inputs = self.process_messages(messages)
        Images=inputs['images']
        # print(Images.shape)

        global GLOBAL_CACHE
        
        # 为了安全，每次生成前清空一下缓存
        GLOBAL_CACHE["spans"] = None
        if not hasattr(self.model, "_is_patched_for_spans"):
            self.model.prepare_inputs_labels_for_multimodal = types.MethodType(custom_prepare_inputs, self.model)
            self.model._is_patched_for_spans = True
            print(">>> [Init] Monkey Patch Applied Successfully!")
        
       #######一堆调试代码#########
        # 生成参数
        generation_kwargs = {
            "input_ids": inputs['input_ids'],
            "images": inputs['images'],
            "do_sample": True if self.temperature > 0 else False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,  # 使用原始值，不调整
            "use_cache": True
            # "stopping_criteria": [stopping_criteria],
        }
        
        # 生成输出
        with torch.inference_mode():
            try:
                output_ids = self.model.generate(**generation_kwargs)
            except Exception as e:
                print(f"生成失败: {e}")
                # if total_input_length > max_context_len:
                #     return f"错误: 输入序列长度({total_input_length})超过模型最大上下文长度({max_context_len})。请减少图片数量或缩短prompt。"
                raise

            modal_token_spans = GLOBAL_CACHE["spans"]
        
        # 解码输出（跳过输入token）
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        # # 移除停止词（如果存在）
        # if generated_text.endswith(stop_str):
        #     generated_text = generated_text[:-len(stop_str)].strip()
        
        return generated_text.strip()
    
    def generate_outputs(self, messages_list):
        """批量生成输出"""
        results = []
        for messages in messages_list:
            result = self.generate_output(messages)
            results.append(result)
        return results
    