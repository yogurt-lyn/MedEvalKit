import torch
import os
import sys
import json
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
        self.feature_repo = {"cls": [], "disease": []}
        
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


    def plot_collected_tsne(self, save_path="auto_tsne.png"):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        cls_list = self.feature_repo["cls"]
        dis_list = self.feature_repo["disease"]
        
        if len(cls_list) < 5:
            print(f"数据太少（当前仅 {len(cls_list)} 组），t-SNE 效果会很差，建议攒够 20 组以上。")
            return

        # 转换数据
        all_cls = torch.cat(cls_list, dim=0).float().numpy()        # [B, H]
        all_dis = torch.cat(dis_list, dim=0).float().numpy()        # [B, 8, H]
        B, _, H = all_dis.shape

        # 准备 t-SNE 输入
        combined = np.concatenate([all_cls, all_dis.reshape(-1, H)], axis=0)
        tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=min(30, B))
        embeds = tsne.fit_transform(combined)
        
        cls_2d = embeds[:B]
        dis_2d = embeds[B:].reshape(B, 8, 2)

        # 绘图逻辑 (参考你提供的图片风格)
        plt.figure(figsize=(10, 10))
        # 我们可以按“疾病种类”上色，这样就能出你图中那种 8 个簇的效果
        colors = plt.cm.get_cmap('Dark2', 8) 
        
        for d_idx in range(8): # 遍历 8 种疾病维度
            d_points = dis_2d[:, d_idx, :] # 所有样本的同一种疾病特征
            plt.scatter(d_points[:, 0], d_points[:, 1], s=15, color=colors(d_idx), alpha=0.6, label=f"Disease {d_idx}")

        # 画 CLS (全局特征) 作为对照
        plt.scatter(cls_2d[:, 0], cls_2d[:, 1], s=50, marker='*', color='black', alpha=0.8, label="Global CLS")

        plt.legend()
        plt.title(f"t-SNE Alignment (Samples: {B})")
        plt.savefig(save_path, dpi=300)
        print(f"自动获取特征完成，t-SNE 已保存至 {save_path}")
    
   
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
    
    def generate_output(self, messages):
        """生成输出"""
        inputs = self.process_messages(messages)
        
        conv = conv_templates[self.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        
        # 设置停止条件
        stopping_criteria = KeywordsStoppingCriteria(
            keywords,  # 使用对话模板的停止词
            self.tokenizer, 
            inputs['input_ids']
        )
        
        # 生成参数
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

        # --- 修正后的特征捕获逻辑 ---
        # 1. 临时开启 train 模式以触发 llava_llama.py 中的诊断逻辑
        self.model.train() 
        with torch.no_grad():
            _ = self.model(input_ids=inputs['input_ids'], images=inputs['images'])
            # 2. 捕获特征并存入仓库
            if hasattr(self.model, 'temp_captured_cls_feat'):
                # 注意：这里建议 detach() 一下，彻底断开梯度流，节省内存
                cls_f = self.model.temp_captured_cls_feat.detach().cpu()
                dis_f = self.model.temp_captured_disease_feats.detach().cpu()
                self.feature_repo["cls"].append(cls_f)
                self.feature_repo["disease"].append(dis_f)
        
        # 3. 必须恢复 eval 模式，否则 generate 可能会报错或行为异常
        self.model.eval() 
        # --- 修正结束 ---
        
        # 生成文本输出
        with torch.inference_mode():
            try:
                output_ids = self.model.generate(**generation_kwargs)
            except Exception as e:
                raise e
        
        # 解码输出
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        if generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)].strip()
        
        return generated_text.strip()
    
    def generate_outputs(self, messages_list):
        """批量生成输出"""
        results = []
        for messages in messages_list:
            result = self.generate_output(messages)
            results.append(result)
        if len(self.feature_repo["cls"]) > 0:
            print(f"检测到 {len(self.feature_repo['cls'])} 组特征，正在绘制逻辑对齐图...")
            self.plot_collected_tsne("ACMMM_NeoRed_Alignment.png")
        return results
    