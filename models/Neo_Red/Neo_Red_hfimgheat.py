import torch
import os
import sys
import json
import math
import cv2
import numpy as np
from PIL import Image

sys.path.append("/data01/kjt_data/MLLM/NeoRed")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images

class Llava:
    def __init__(self, model_path, args):
        super().__init__()
        
        print(f"Loading LLaVA-Rad model from: {model_path}")
        model_base = os.path.join("/data_lyn/LLaVARadWeightNeoRed", "vicuna-7b-v1.5")
        
        # 禁用torch初始化优化（LLaVA推荐）
        disable_torch_init()
        # 使用官方库的load_pretrained_model函数加载模型
        try:
            # 使用 "llavarad" 作为模型名称
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
        
        # 设置生成参数
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        
        # 设置对话模板（LLaVA-Rad使用v1格式）
        self.conv_mode = "v1"
    
    def process_messages(self, messages):
        """处理输入消息，转换为LLaVA格式"""
        prompt = ""
        images = None
        
        # 处理系统提示词
        if "system" in messages:
            prompt = messages["system"]
        
        # 添加用户提示词
        if "prompt" in messages:
            if prompt:
                prompt = prompt + "\n" + messages["prompt"]
            else:
                prompt = messages["prompt"]
        
        # 处理图像
        if "images" in messages:
            images = messages["images"]
            if not isinstance(images, list):
                images = [images]
            images = [img if isinstance(img, Image.Image) else Image.open(img).convert('RGB') for img in images]
        elif "image" in messages:
            image = messages["image"]
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                image = image.convert('RGB')
            images = [image]
        else:
            images = None
        
        # 使用LLaVA的对话模板
        conv = conv_templates[self.conv_mode].copy()
        
        query = prompt 
        
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # 处理图像
        if images:
            image_tensor = process_images(images, self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        else:
            image_tensor = None
        
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

    def generate_output(self, messages, output_attention_path=None):
        """生成输出并可选保存注意力热力图"""
        inputs = self.process_messages(messages)
        
        conv = conv_templates[self.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, 
            self.tokenizer, 
            inputs['input_ids']
        )
        
        generation_kwargs = {
            "input_ids": inputs['input_ids'],
            "images": inputs['images'],
            "do_sample": True if self.temperature > 0 else False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "stopping_criteria": [stopping_criteria],
        }
        
        # [新增] 如果需要可视化注意力，让模型返回 attention
        if output_attention_path is not None:
            generation_kwargs.update({
                "output_attentions": True,
                "return_dict_in_generate": True,
            })
        
        with torch.inference_mode():
            try:
                outputs = self.model.generate(**generation_kwargs)
            except Exception as e:
                raise
        
        # 处理返回结果
        if output_attention_path is not None:
            output_ids = outputs.sequences
        else:
            output_ids = outputs
            
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0]

        if generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)].strip()
            
        # ========================================================
        # 注意力提取与可视化逻辑
        # ========================================================
        if output_attention_path is not None and inputs['images'] is not None:
            try:
                image_token_pos = (inputs['input_ids'][0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                if len(image_token_pos) > 0:
                    image_start_idx = image_token_pos[0].item()
                    
                    first_token_attns = outputs.attentions[0] 
                    last_layer_attn = first_token_attns[-1] 
                    expanded_prompt_len = last_layer_attn.shape[-1]
                    
                    orig_len = inputs['input_ids'].shape[1]
                    num_image_tokens = expanded_prompt_len - orig_len + 1
                    
                    total_img_attn = torch.zeros(num_image_tokens, device=self.model.device)
                    num_generated_tokens = len(outputs.attentions)
                    
                    for i in range(num_generated_tokens):
                        layer_attn = outputs.attentions[i][-1] 
                        
                        if i == 0:
                            attn = layer_attn[0, :, -1, :] 
                        else:
                            attn = layer_attn[0, :, 0, :] 
                        
                        attn_mean_heads = attn.mean(dim=0)
                        img_attn = attn_mean_heads[image_start_idx : image_start_idx + num_image_tokens]
                        total_img_attn += img_attn
                        
                    total_img_attn /= num_generated_tokens
                    
                    grid_size = int(math.sqrt(num_image_tokens))
                    attn_2d = total_img_attn.view(grid_size, grid_size).float().cpu().numpy()
                    
                    # 归一化
                    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
                    
                    # 读取原图
                    orig_image = messages.get("images", messages.get("image"))
                    if isinstance(orig_image, list):
                        orig_image = orig_image[0]
                        
                    if isinstance(orig_image, str):
                        orig_image_cv = cv2.imread(orig_image)
                    elif isinstance(orig_image, Image.Image):
                        orig_image_cv = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
                    else:
                        raise ValueError("无法解析原始图像类型")
                        
                    # 生成热力图
                    heatmap = cv2.resize(attn_2d, (orig_image_cv.shape[1], orig_image_cv.shape[0]))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    # 融合并保存
                    superimposed_img = cv2.addWeighted(orig_image_cv, 0.6, heatmap, 0.4, 0)
                    cv2.imwrite(output_attention_path, superimposed_img)
                    # print(f"[*] 注意力热力图保存至: {output_attention_path}")
            except Exception as e:
                print(f"[!] 注意力可视化失败: {e}")

        return generated_text.strip()

    def generate_outputs(self, messages_list, output_dir="/data_lyn/llavaradheat"):
        """批量生成输出"""
        results = []
        for idx, messages in enumerate(messages_list):
            attn_path = os.path.join(output_dir, f"heatmap_{idx}.png") if output_dir else None
            result = self.generate_output(messages, output_attention_path=attn_path)
            results.append(result)
        return results