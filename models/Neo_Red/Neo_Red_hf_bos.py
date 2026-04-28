# import numpy as np
# import torch
# import os
# import sys
# import json
# from PIL import Image
# sys.path.append("/data01/kjt_data/MLLM/NeoRed")

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images

# class Llava:
#     def __init__(self, model_path, args):
#         super().__init__()
        
#         print(f"Loading LLaVA-Rad model from: {model_path}")
#         model_base = os.path.join("/data_lyn/LLaVARadWeightNeoRed", "vicuna-7b-v1.5")
        
#         # 禁用torch初始化优化（LLaVA推荐）
#         disable_torch_init()
#         # 使用官方库的load_pretrained_model函数加载模型
#         try:
#             # 使用 "llavarad" 作为模型名称
#             self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
#                 model_path=model_path,
#                 model_base=model_base,
#                 model_name="llavarad",
#                 load_8bit=False,
#                 load_4bit=False,
#                 device_map="auto"
#             )
#             print("LLaVA-Rad model loaded successfully!")
            
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             import traceback
#             traceback.print_exc()
#             raise
        
#         # 设置生成参数
#         self.temperature = args.temperature
#         self.top_p = args.top_p
#         self.repetition_penalty = args.repetition_penalty
#         self.max_new_tokens = args.max_new_tokens
        
#         # 设置对话模板（LLaVA-Rad使用v1格式）
#         self.conv_mode = "v1"
    
   
#     def process_messages(self, messages):
#         """处理输入消息，转换为LLaVA格式"""
#         prompt = ""
#         images = None
        
#         # 处理系统提示词
#         if "system" in messages:
#             prompt = messages["system"]
        
#         # 添加用户提示词
#         if "prompt" in messages:
#             if prompt:
#                 prompt = prompt + "\n" + messages["prompt"]
#             else:
#                 prompt = messages["prompt"]
        
#         # 处理图像
#         if "images" in messages:
#             images = messages["images"]
#             if not isinstance(images, list):
#                 images = [images]
#             # 确保图像是PIL Image对象
#             images = [img if isinstance(img, Image.Image) else Image.open(img).convert('RGB') for img in images]
#         elif "image" in messages:
#             image = messages["image"]
#             if isinstance(image, str):
#                 image = Image.open(image).convert('RGB')
#             elif not isinstance(image, Image.Image):
#                 image = image.convert('RGB')
#             images = [image]
#         else:
#             images = None
        
#         # # 使用LLaVA的对话模板
#         conv = conv_templates[self.conv_mode].copy()
        
#         # # 构建查询：如果有多张图像，在prompt中为每张图像添加<image>标记
#         # if images:
#         #     # 每张图片用换行分隔
#         #     if len(images) == 1:
#         #         image_tokens = DEFAULT_IMAGE_TOKEN
#         #     else:
#         #         # 多张图片：每张图片占一行
#         #         image_tokens = "\n".join([DEFAULT_IMAGE_TOKEN] * len(images))
            
#         #     if prompt:
#         #         query = f"{image_tokens}\n{prompt}"
#         #     else:
#         #         query = image_tokens
#         # else:
#         #     query = prompt if prompt else ""
        
#         query = prompt 
        
#         conv.append_message(conv.roles[0], query)
#         conv.append_message(conv.roles[1], None)
#         prompt_text = conv.get_prompt()
        
#         # 处理图像
#         if images:
#             # 使用官方库的process_images函数
#             image_tensor = process_images(images, self.image_processor, self.model.config)
#             if type(image_tensor) is list:
#                 image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
#             else:
#                 image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
#         else:
#             image_tensor = None
        
#         # 使用官方库的tokenizer_image_token函数处理prompt
#         input_ids = tokenizer_image_token(
#             prompt_text, 
#             self.tokenizer, 
#             IMAGE_TOKEN_INDEX, 
#             return_tensors='pt'
#         ).unsqueeze(0).to(self.model.device)
        
#         return {
#             'input_ids': input_ids,
#             'images': image_tensor
#         }
    
#     def plot_attention_heatmap(self, matrix, x_tokens, y_tokens, save_path):
#         """绘制并保存热力图"""
#         plt.figure(figsize=(24, 12))
        
#         # 使用 seaborn 绘制。注意：矩阵行是生成的 token，列是历史 token
#         sns.heatmap(
#             matrix, 
#             xticklabels=x_tokens, 
#             yticklabels=y_tokens, 
#             cmap="viridis",
#             annot=False,
#             cbar_kws={'label': 'Attention Score'}
#         )
        
#         plt.title(f"Attention Analysis: Generated Tokens vs History (Focus on <s>)")
#         plt.xlabel("History Tokens (Keys)")
#         plt.ylabel("Generated Tokens (Queries)")
#         plt.xticks(rotation=90, fontsize=8)
#         plt.yticks(fontsize=8)
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300)
#         plt.close()
#         print(f">>> Attention heatmap saved to: {save_path}")

#     import torch
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import os

#     def generate_output(self, messages, sample_idx=0, visualize=True):
#         inputs = self.process_messages(messages)
#         input_ids = inputs['input_ids']
#         input_len = input_ids.shape[1]
        
#         generation_kwargs = {
#             "input_ids": input_ids,
#             "images": inputs['images'],
#             "do_sample": True if self.temperature > 0 else False,
#             "temperature": self.temperature,
#             "top_p": self.top_p,
#             "repetition_penalty": self.repetition_penalty,
#             "max_new_tokens": self.max_new_tokens,
#             "use_cache": True,
#             "output_attentions": True,       
#             "return_dict_in_generate": True, 
#         }

#         with torch.inference_mode():
#             outputs = self.model.generate(**generation_kwargs)
        
#         # 1. 提取生成的 Token 并解码文本
#         output_ids = outputs.sequences[0]
#         gen_ids = output_ids[input_len:]
#         generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

#         # --- 注意力提取逻辑 ---
#         if visualize and hasattr(outputs, "attentions"):
#             # outputs.attentions 结构: (num_gen_tokens, num_layers, batch, num_heads, 1, current_seq_len)
#             num_gen_steps = len(outputs.attentions)
#             num_layers = len(outputs.attentions[0])
            
#             # 初始化统计数组
#             bos_attn_per_layer = np.zeros(num_layers)
#             others_attn_per_layer = np.zeros(num_layers)

#             for step in range(num_gen_steps):
#                 for layer in range(num_layers):
#                     # 提取当前步、当前层的注意力 [num_heads, 1, current_seq_len]
#                     # 注意：Llava/HuggingFace 在 generate 模式下返回的是最后一块的 attention
#                     layer_attn = outputs.attentions[step][layer][0].mean(dim=0).cpu() 
                    
#                     current_seq_len = layer_attn.shape[-1]
                    
#                     # 提取对 BOS (index 0) 的注意力
#                     bos_val = layer_attn[0, 0].item()
                    
#                     # 提取对其他 Token 的平均注意力
#                     if current_seq_len > 1:
#                         others_val = layer_attn[0, 1:].mean().item()
#                     else:
#                         others_val = 0.0
                    
#                     bos_attn_per_layer[layer] += bos_val
#                     others_attn_per_layer[layer] += others_val

#             # 计算生成全过程的平均值
#             bos_attn_per_layer /= num_gen_steps
#             others_attn_per_layer /= num_gen_steps

#             # 调用绘图函数
#             self.plot_layer_attention_curve(bos_attn_per_layer, others_attn_per_layer, sample_idx)

#         # 最终返回生成的文本
#         return generated_text.strip()

#     def plot_layer_attention_curve(self, bos_scores, others_scores, sample_idx):
#         plt.figure(figsize=(10, 6))
#         layers = np.arange(1, len(bos_scores) + 1)
#         plt.yscale('log')

#         # 绘制蓝色曲线：BOS (锚点)
#         plt.plot(layers, bos_scores, label='Average Attention to <BOS>', 
#                 color='#1f77b4', marker='o', linewidth=2, markersize=4)
        
#         # 绘制灰色曲线：其他所有 Token 的均值
#         plt.plot(layers, others_scores, label='Average Attention to Other Tokens', 
#                 color='#d3d3d3', linestyle='--', linewidth=2)

#         plt.title(f'Layer-wise Attention Anchor Analysis (Sample {sample_idx})', fontsize=14)
#         plt.xlabel('Transformer Layer', fontsize=12)
#         plt.ylabel('Mean Attention Score', fontsize=12)
#         plt.grid(True, linestyle=':', alpha=0.6)
#         plt.legend()

#         os.makedirs("attention_curves", exist_ok=True)
#         plt.savefig(f"attention_curves/layer_dist_{sample_idx}.png", dpi=300)
#         plt.close()

#     def generate_outputs(self, messages_list):
#         """批量生成输出"""
#         results = []
#         for idx, messages in enumerate(messages_list):
#             print(f"Processing sample {idx+1}/{len(messages_list)}")
#             # 传入 idx 以便区分保存的图片文件名
#             result = self.generate_output(messages, sample_idx=idx)
#             results.append(result)
#         return results


#####不画图 存数据版

import numpy as np
import torch
import os
import sys
import json
from PIL import Image
sys.path.append("/data01/kjt_data/MLLM/NeoRed")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images

class Llava:
    def __init__(self, model_path, args):
        super().__init__()
        
        print(f"Loading LLaVA-Rad model from: {model_path}")
        model_base = os.path.join("/data01/kjt_data/MLLM/LLaVARadWeight", "vicuna-7b-v1.5")
        
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
        
        # # 使用LLaVA的对话模板
        conv = conv_templates[self.conv_mode].copy()
        
        # # 构建查询：如果有多张图像，在prompt中为每张图像添加<image>标记
        # if images:
        #     # 每张图片用换行分隔
        #     if len(images) == 1:
        #         image_tokens = DEFAULT_IMAGE_TOKEN
        #     else:
        #         # 多张图片：每张图片占一行
        #         image_tokens = "\n".join([DEFAULT_IMAGE_TOKEN] * len(images))
            
        #     if prompt:
        #         query = f"{image_tokens}\n{prompt}"
        #     else:
        #         query = image_tokens
        # else:
        #     query = prompt if prompt else ""
        
        query = prompt 
        
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
    
    def plot_attention_heatmap(self, matrix, x_tokens, y_tokens, save_path):
        """绘制并保存热力图"""
        plt.figure(figsize=(24, 12))
        
        # 使用 seaborn 绘制。注意：矩阵行是生成的 token，列是历史 token
        sns.heatmap(
            matrix, 
            xticklabels=x_tokens, 
            yticklabels=y_tokens, 
            cmap="viridis",
            annot=False,
            cbar_kws={'label': 'Attention Score'}
        )
        
        plt.title(f"Attention Analysis: Generated Tokens vs History (Focus on <s>)")
        plt.xlabel("History Tokens (Keys)")
        plt.ylabel("Generated Tokens (Queries)")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f">>> Attention heatmap saved to: {save_path}")

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    def generate_output(self, messages, sample_idx=0, visualize=True):
        inputs = self.process_messages(messages)
        input_ids = inputs['input_ids']
        input_len = input_ids.shape[1]
        
        generation_kwargs = {
            "input_ids": input_ids,
            "images": inputs['images'],
            "do_sample": True if self.temperature > 0 else False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "output_attentions": True,       
            "return_dict_in_generate": True, 
        }

        with torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs)
        
        output_ids = outputs.sequences[0]
        gen_ids = output_ids[input_len:]
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # --- 注意力数据提取与保存 ---
        if visualize and hasattr(outputs, "attentions"):
            num_gen_steps = len(outputs.attentions)
            num_layers = len(outputs.attentions[0])
            
            bos_attn_per_layer = np.zeros(num_layers)
            others_attn_per_layer = np.zeros(num_layers)

            for step in range(num_gen_steps):
                for layer in range(num_layers):
                    # 获取该步该层的注意力 [num_heads, 1, current_seq_len]
                    layer_attn = outputs.attentions[step][layer][0].mean(dim=0).cpu() 
                    current_seq_len = layer_attn.shape[-1]
                    
                    # 提取 BOS 注意力
                    bos_val = layer_attn[0, 0].item()
                    
                    # 提取其他 Token 平均注意力
                    if current_seq_len > 1:
                        others_val = layer_attn[0, 1:].mean().item()
                    else:
                        others_val = 0.0
                    
                    bos_attn_per_layer[layer] += bos_val
                    others_attn_per_layer[layer] += others_val

            # 计算平均值
            bos_attn_per_layer /= num_gen_steps
            others_attn_per_layer /= num_gen_steps

            # --- 改为保存 JSON ---
            self.save_attention_json(
                bos_scores=bos_attn_per_layer, 
                others_scores=others_attn_per_layer, 
                sample_idx=sample_idx,
                text=generated_text.strip()
            )

        return generated_text.strip()
    
    def save_attention_json(self, bos_scores, others_scores, sample_idx, text):
        """将所有样本的注意力数据汇总保存到同一个 JSON 文件中"""
        file_path = "all_samples_attention_databos_neored.json"
        
        # 1. 构造当前样本的数据字典
        new_sample_data = {
            "sample_idx": sample_idx,
            "generated_text": text,
            "layers": list(range(1, len(bos_scores) + 1)),
            "bos_attention_scores": bos_scores.tolist(),
            "others_attention_scores": others_scores.tolist()
        }

        # 2. 读取现有数据（如果文件存在）
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
                    if not isinstance(all_data, list):
                        all_data = []
            except (json.JSONDecodeError, IOError):
                all_data = []
        else:
            all_data = []

        # 3. 将新数据追加到列表
        all_data.append(new_sample_data)

        # 4. 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
            
        print(f">>> Sample {sample_idx} aggregated into: {file_path}")

    # def save_attention_json(self, bos_scores, others_scores, sample_idx, text):
    #     """将注意力数据保存为 JSON 文件"""
    #     # 注意：NumPy 数组不能直接序列化，需要转为 list
    #     data = {
    #         "sample_idx": sample_idx,
    #         "generated_text": text,
    #         "layers": list(range(1, len(bos_scores) + 1)),
    #         "bos_attention_scores": bos_scores.tolist(),
    #         "others_attention_scores": others_scores.tolist()
    #     }

    #     save_dir = "attention_data_json"
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     file_path = os.path.join(save_dir, f"sample_{sample_idx}.json")
        
    #     with open(file_path, 'w', encoding='utf-8') as f:
    #         json.dump(data, f, indent=4, ensure_ascii=False)
            
    #     print(f">>> Attention data saved to: {file_path}")

    def generate_outputs(self, messages_list):
        """批量生成输出"""
        results = []
        for idx, messages in enumerate(messages_list):
            print(f"Processing sample {idx+1}/{len(messages_list)}")
            # 传入 idx 以便区分保存的图片文件名
            result = self.generate_output(messages, sample_idx=idx)
            results.append(result)
        return results
    