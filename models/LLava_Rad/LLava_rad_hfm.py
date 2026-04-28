import torch
import os
import sys
import json
from PIL import Image
sys.path.append("/media/hdd2/MedEvalKit/LLaVARad")


from LLaVARad.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from LLaVARad.llava.conversation import conv_templates, SeparatorStyle
from LLaVARad.llava.model.builder import load_pretrained_model
from LLaVARad.llava.utils import disable_torch_init
from LLaVARad.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images


class Llava:
    def __init__(self, model_path, args):
        super().__init__()
        
        print(f"Loading LLaVA-Rad model from: {model_path}")
        model_base = os.path.join("/media/hdd2/MedEvalKit/LLaVARadWeight", "vicuna-7b-v1.5")
        
        # 禁用torch初始化优化（LLaVA推荐）
        disable_torch_init()
        # 使用官方库的load_pretrained_model函数加载模型
        try:
            # 使用 "llavarad" 作为模型名称
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=model_base,
                model_name="llava_rad",
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
            # 确保图像是PIL Image对象
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
        
        # 构建查询：如果有多张图像，在prompt中为每张图像添加<image>标记
        if images:
            # 每张图片用换行分隔
            if len(images) == 1:
                image_tokens = DEFAULT_IMAGE_TOKEN
            else:
                # 多张图片：每张图片占一行
                image_tokens = "\n".join([DEFAULT_IMAGE_TOKEN] * len(images))
            
            if prompt:
                query = f"{image_tokens}\n{prompt}"
            else:
                query = image_tokens
        else:
            query = prompt if prompt else ""
        
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # 处理图像
        if images:
            # 使用官方库的process_images函数
            image_tensor = process_images(images, self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        else:
            image_tensor = None
        
        # 使用官方库的tokenizer_image_token函数处理prompt
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
        
        # ========== 增强的调试：详细检查输入序列长度 ==========
        # input_ids_len = inputs['input_ids'].shape[1]
        
        # # 更准确地获取图片数量
        # if "images" in messages:
        #     num_images = len(messages["images"]) if isinstance(messages["images"], list) else 1
        # elif "image" in messages:
        #     num_images = 1
        # else:
        #     num_images = 0
        
        # # 获取模型的最大上下文长度
        # max_context_len = getattr(self.model.config, 'max_position_embeddings', None) or \
        #                  getattr(self.model.config, 'max_sequence_length', None) or \
        #                  getattr(self.model.config, 'tokenizer_model_max_length', None) or \
        #                  self.context_len if hasattr(self, 'context_len') else 4096
        
        # # 详细计算图片特征占用的token数量
        # image_tokens_count = 0
        # num_patches_per_image = 0
        # image_shape_info = "N/A"
        
        # if inputs['images'] is not None:
        #     try:
        #         vision_tower = self.model.get_vision_tower()
                
        #         # 获取图片的实际 shape
        #         if isinstance(inputs['images'], list):
        #             image_shape_info = f"List of {len(inputs['images'])} tensors"
        #             if len(inputs['images']) > 0:
        #                 first_img_shape = inputs['images'][0].shape if hasattr(inputs['images'][0], 'shape') else "N/A"
        #                 image_shape_info += f", first shape: {first_img_shape}"
        #         else:
        #             image_shape_info = str(inputs['images'].shape)
                
        #         # 获取 num_patches
        #         if vision_tower is not None:
        #             # 尝试多种方式获取 num_patches
        #             if hasattr(vision_tower, 'num_patches'):
        #                 num_patches_per_image = vision_tower.num_patches
        #             elif hasattr(vision_tower, 'vision_tower') and hasattr(vision_tower.vision_tower, 'num_patches'):
        #                 num_patches_per_image = vision_tower.vision_tower.num_patches
        #             else:
        #                 # 尝试从图片 shape 推断
        #                 if isinstance(inputs['images'], list) and len(inputs['images']) > 0:
        #                     img_tensor = inputs['images'][0]
        #                     if hasattr(img_tensor, 'shape') and len(img_tensor.shape) >= 2:
        #                         # 假设 shape 是 [batch, num_patches, hidden_dim] 或类似
        #                         num_patches_per_image = img_tensor.shape[1] if img_tensor.shape[1] > 0 else 0
        #                 elif hasattr(inputs['images'], 'shape'):
        #                     if len(inputs['images'].shape) >= 2:
        #                         num_patches_per_image = inputs['images'].shape[1] if inputs['images'].shape[1] > 0 else 0
                    
        #             # 计算总图片 token 数量
        #             if isinstance(inputs['images'], list):
        #                 image_tokens_count = len(inputs['images']) * num_patches_per_image
        #             else:
        #                 if inputs['images'].ndim >= 2:
        #                     # 检查是否是 batch 维度
        #                     if inputs['images'].shape[0] > 1:
        #                         image_tokens_count = inputs['images'].shape[0] * num_patches_per_image
        #                     else:
        #                         image_tokens_count = num_patches_per_image
        #                 else:
        #                     image_tokens_count = num_patches_per_image
                            
        #         else:
        #             # 如果无法获取 vision_tower，使用估算值
        #             num_patches_per_image = 1369  # 基于之前的调试输出
        #             image_tokens_count = num_images * num_patches_per_image
        #             print(f"Warning: 无法获取 vision_tower，使用估算值: {num_patches_per_image} tokens/图片")
                    
        #     except Exception as e:
        #         # 如果无法获取，使用估算值
        #         num_patches_per_image = 1369
        #         image_tokens_count = num_images * num_patches_per_image
        #         print(f"Warning: 获取图片 token 数量时出错: {e}")
        #         print(f"使用估算值: {num_patches_per_image} tokens/图片")
        
        # # 计算文本 prompt 的 token 数量（不包括图片占位符）
        # # 统计 prompt 中的 <image> token 数量
        # prompt_text = ""
        # if "prompt" in messages:
        #     prompt_text = messages["prompt"]
        
        # # 计算总输入长度
        # total_input_length = input_ids_len + image_tokens_count
        
        # # 详细打印调试信息
        # print(f"\n{'='*70}")
        # print(f"输入序列长度详细分析")
        # print(f"{'='*70}")
        # print(f"图片信息:")
        # print(f"   图片数量: {num_images}")
        # print(f"   图片 tensor shape: {image_shape_info}")
        # print(f"   每张图片的 patches 数量: {num_patches_per_image}")
        # print(f"   图片特征总 token 数量: {image_tokens_count}")
        # print(f"   平均每张图片占用: {image_tokens_count // num_images if num_images > 0 else 0} tokens")
        # print(f"")
        # print(f"文本信息:")
        # print(f"   文本 token 数量 (input_ids): {input_ids_len}")
        # if prompt_text:
        #     prompt_words = len(prompt_text.split())
        #     print(f"   Prompt 文本长度: {len(prompt_text)} 字符, ~{prompt_words} 单词")
        # print(f"")
        # print(f"总长度统计:")
        # print(f"   总输入长度: {total_input_length} tokens")
        # print(f"   模型最大上下文长度: {max_context_len} tokens")
        # print(f"   使用率: {total_input_length / max_context_len * 100:.1f}%")
        # print(f"   可用 token 空间: {max_context_len - total_input_length} tokens")
        # print(f"")
        
        # if total_input_length > max_context_len:
        #     print(f"严重警告: 超过限制 {total_input_length - max_context_len} tokens!")
        #     print(f"这可能导致模型截断输入或生成失败")
        #     print(f"建议: 减少图片数量或缩短 prompt")
        # else:
        #     print(f"未超过限制，安全")
        #     remaining_for_generation = max_context_len - total_input_length
        #     print(f"可用于生成的最大 token 数: {remaining_for_generation}")
        
        # print(f"")
        # print(f"生成参数:")
        # print(f"原始 max_new_tokens 设置: {self.max_new_tokens}")
        # print(f"{'='*70}\n")
        # ========== 调试结束 ==========
        
        # 使用对话模板的停止词（参考官方示例）
        conv = conv_templates[self.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        
        # 设置停止条件
        stopping_criteria = KeywordsStoppingCriteria(
            keywords,  # 使用对话模板的停止词
            self.tokenizer, 
            inputs['input_ids']
        )
        
        # # 不调整max_new_tokens，使用原始设置，让模型自由输出
        # if total_input_length > max_context_len:
        #     print(f"警告: 输入序列超过限制，但保持 max_new_tokens={self.max_new_tokens}，让模型自由输出")
        #     print(f"如果生成失败，可能需要减少图片数量或缩短 prompt")
        
        # 生成参数
        generation_kwargs = {
            "input_ids": inputs['input_ids'],
            "images": inputs['images'],
            "do_sample": True if self.temperature > 0 else False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,  # 使用原始值，不调整
            "use_cache": True,
            "stopping_criteria": [stopping_criteria],
        }
        
        # 生成输出
        with torch.inference_mode():
            try:
                output_ids = self.model.generate(**generation_kwargs)
            except Exception as e:
                # print(f"生成失败: {e}")
                # if total_input_length > max_context_len:
                #     return f"错误: 输入序列长度({total_input_length})超过模型最大上下文长度({max_context_len})。请减少图片数量或缩短prompt。"
                raise
        
        # 解码输出（跳过输入token）
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        # 移除停止词（如果存在）
        if generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)].strip()
        
        return generated_text.strip()
    
    def generate_outputs(self, messages_list):
        """批量生成输出"""
        results = []
        for messages in messages_list:
            result = self.generate_output(messages)
            results.append(result)
        return results