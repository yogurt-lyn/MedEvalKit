import torch
import os
import sys
import json
import types
import gc
from PIL import Image

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/data01/kjt_data/MLLM/NeoRed")

from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, 
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images

# ==========================================
# 1. 注意力可视化的全局变量和 Helper 函数
# ==========================================
GLOBAL_CACHE = {"spans": None}

MARKER_ID_PATTERNS = {
    "dev": {
        "start": [29966, 3359, 29958],  
        "end":   [829, 3359, 29958],
    },
    "peri": {
        "start": [29966, 546, 29875, 29958],
        "end":   [829, 546, 29875, 29958],
    },
    "phys": {
        "start": [29966, 14017, 29958],
        "end":   [829, 14017, 29958],
    },
}

def find_subseq_ids(seq, pattern):
    m = len(pattern)
    for i in range(len(seq) - m + 1):
        if seq[i:i + m] == pattern:
            return i, i + m
    return None

def generate_I_D_mask_from_labels(input_ids, labels):
    if labels is None:
        return None, None
        
    bsz, seqlen = labels.shape
    I_mask = torch.zeros_like(labels, dtype=torch.bool)
    D_mask = torch.zeros_like(labels, dtype=torch.bool)

    IMAGING_CONCLUSION_IDS = [6751, 15997, 29901]
    DISEASE_DIAGNOSIS_IDS  = [559, 24876, 19263, 29901]

    for b in range(bsz):
        label = labels[b]
        valid_pos = (label != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        if len(valid_pos) == 0:
            continue
            
        start, end = valid_pos[0].item(), valid_pos[-1].item() + 1
        label_ids = input_ids[b, start:end].tolist()

        im_start = None
        for i in range(len(label_ids) - len(IMAGING_CONCLUSION_IDS) + 1):
            if label_ids[i:i+len(IMAGING_CONCLUSION_IDS)] == IMAGING_CONCLUSION_IDS:
                im_start = i
                break

        d_start = None
        for i in range(len(label_ids) - len(DISEASE_DIAGNOSIS_IDS) + 1):
            if label_ids[i:i+len(DISEASE_DIAGNOSIS_IDS)] == DISEASE_DIAGNOSIS_IDS:
                d_start = i
                break

        if im_start is None or d_start is None:
            continue

        im_content_start = start - 3 + im_start + len(IMAGING_CONCLUSION_IDS)
        im_content_end   = start + d_start - 1 
    
        d_content_start = start - 6 + d_start + len(DISEASE_DIAGNOSIS_IDS)
        d_content_end   = end

        I_mask[b, im_content_start:im_content_end] = True
        D_mask[b, d_content_start:d_content_end] = True
        
    I_mask = [I_mask[b] for b in range(bsz)]
    D_mask = [D_mask[b] for b in range(bsz)]
    return I_mask, D_mask

# LLaVA 的 Monkey Patch，用于截获 Token Span
def custom_prepare_inputs(self, input_ids, attention_mask, past_key_values, labels, images):
    vision_tower = self.get_vision_tower()
    
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
             attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
        return input_ids, attention_mask, past_key_values, None, labels, None, None, None

    if type(images) is list or images.ndim == 5:
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        image_features = [x.flatten(0, 1) for x in image_features]
    else:
        image_features = self.encode_images(images)

    new_input_embeds = []
    new_labels = [] if labels is not None else None
    cur_image_idx = 0
    modal_token_spans = []

    I_mask, D_mask = generate_I_D_mask_from_labels(input_ids, labels)

    for batch_idx, cur_input_ids in enumerate(input_ids):
        if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
            half_len = cur_input_ids.shape[0] // 2
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
            cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
        
        image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        cur_new_input_embeds = []
        if labels is not None:
            cur_labels = labels[batch_idx]
            cur_new_labels = []
            
        while image_token_indices.numel() > 0:
            cur_image_features = image_features[cur_image_idx]
            image_token_start = image_token_indices[0]
            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                    cur_labels = cur_labels[image_token_start+2:]
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]
            cur_image_idx += 1
            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                cur_input_ids = cur_input_ids[image_token_start+2:]
            else:
                cur_input_ids = cur_input_ids[image_token_start+1:]
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
        if cur_input_ids.numel() > 0:
            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
            if labels is not None:
                cur_new_labels.append(cur_labels)
        
        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        new_input_embeds.append(cur_new_input_embeds)
        if labels is not None:
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

        cur_input_ids_raw = input_ids[batch_idx]
        cur_input_ids_tmp = cur_input_ids_raw.clone()

        spans = {
            "image": [],
            "dev": None,
            "peri": None,
            "phys": None
        }

        cur_pos = 0
        cur_image_idx_for_span = 0

        num_images = (cur_input_ids_raw == IMAGE_TOKEN_INDEX).sum().item()
        total_offset = 0
        if num_images > 0:
            img_len = image_features[0].shape[0]
            total_offset = num_images * (img_len - 1)

        if total_offset > 0 and I_mask is not None and D_mask is not None:
            cur_I_mask = I_mask[batch_idx]
            cur_D_mask = D_mask[batch_idx]
            new_len = cur_I_mask.shape[0] + total_offset
            shifted_I = torch.zeros(new_len, dtype=cur_I_mask.dtype, device=cur_I_mask.device)
            shifted_D = torch.zeros(new_len, dtype=cur_D_mask.dtype, device=cur_D_mask.device)
            shifted_I[total_offset:] = cur_I_mask
            shifted_D[total_offset:] = cur_D_mask
            I_mask[batch_idx] = shifted_I
            D_mask[batch_idx] = shifted_D

        while (cur_input_ids_tmp == IMAGE_TOKEN_INDEX).any():
            image_indices = torch.where(cur_input_ids_tmp == IMAGE_TOKEN_INDEX)[0]
            token_idx = image_indices[0].item()
            start = cur_pos + token_idx
            img_len = image_features[cur_image_idx_for_span].shape[0]
            end = start + img_len
            spans["image"].append((start, end))
            cur_pos = end
            cur_input_ids_tmp = cur_input_ids_tmp[token_idx + 1:]
            cur_image_idx_for_span += 1

        if len(spans["image"]) > 0:
            spans["image"] = (spans["image"][0][0], spans["image"][-1][1])
        else:
            spans["image"] = None

        input_id_list = cur_input_ids_raw.tolist()
        for key, pat in MARKER_ID_PATTERNS.items():
            start_pos = find_subseq_ids(input_id_list, pat["start"])
            end_pos   = find_subseq_ids(input_id_list, pat["end"])
            if start_pos is None or end_pos is None:
                spans[key] = None
                continue
            start_marker_end = start_pos[1]
            end_marker_start = end_pos[0]
            if start_marker_end < end_marker_start:
                spans[key] = (
                    start_marker_end + total_offset,
                    end_marker_start + total_offset
                )
            else:
                spans[key] = None

        modal_token_spans.append(spans)

    if input_ids.shape[1] > 1:
        global GLOBAL_CACHE
        GLOBAL_CACHE["spans"] = modal_token_spans

    if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
        max_len = max(x.shape[0] for x in new_input_embeds)
        new_input_embeds_align = []
        for cur_new_embed in new_input_embeds:
            cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
            new_input_embeds_align.append(cur_new_embed)
        new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

        if labels is not None:
            new_labels_align = []
            for cur_new_label in new_labels:
                cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                new_labels_align.append(cur_new_label)
            new_labels = torch.stack(new_labels_align, dim=0)

        if attention_mask is not None:
            new_attention_mask = []
            for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, new_labels, new_labels):
                new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                new_attention_mask.append(cur_new_attention_mask)
            attention_mask = torch.stack(new_attention_mask, dim=0)

        if I_mask is not None and D_mask is not None:
            I_mask_align = []
            for cur_I_mask in I_mask:
                cur_I_mask = torch.cat((cur_I_mask, torch.zeros((max_len - cur_I_mask.shape[0],), dtype=cur_I_mask.dtype, device=cur_I_mask.device)), dim=0)
                I_mask_align.append(cur_I_mask)
            I_mask = torch.stack(I_mask_align, dim=0)
            D_mask_align = []
            for cur_D_mask in D_mask:
                cur_D_mask = torch.cat((cur_D_mask, torch.zeros((max_len - cur_D_mask.shape[0],), dtype=cur_D_mask.dtype, device=cur_D_mask.device)), dim=0)
                D_mask_align.append(cur_D_mask)
            D_mask = torch.stack(D_mask_align, dim=0)
    else:
        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)
        if I_mask is not None and D_mask is not None:
            I_mask = torch.stack(I_mask, dim=0)
            D_mask = torch.stack(D_mask, dim=0)
        if attention_mask is not None:
            new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)

    return None, attention_mask, past_key_values, new_input_embeds, new_labels, modal_token_spans, I_mask, D_mask


# ==========================================
# 2. 原主类及新增的分析方法
# ==========================================
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

    def move_image_token_to_front(self, text: str) -> str:
        """
        将文本中已有的 <image> token 全部移到最前面（顺序不变）
        不新增、不删除
        """
        image_token = DEFAULT_IMAGE_TOKEN

        if image_token not in text:
            return text

        parts = text.split(image_token)
        num_images = len(parts) - 1

        # 去掉原位置的 image token
        text_wo_image = "".join(parts).strip()

        # 前置 image token（数量保持一致）
        image_prefix = "".join([image_token] * num_images)

        if text_wo_image:
            return f"{image_prefix}\n{text_wo_image}"
        else:
            return image_prefix
    
   
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
        
        conv = conv_templates[self.conv_mode].copy()
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

        # 【新增】：为自定义 inputs 生成 attention_mask
        attention_mask = torch.ones_like(input_ids).to(self.model.device)
        
        return {
            'input_ids': input_ids,
            'images': image_tensor,
            'attention_mask': attention_mask # 添加 mask 用于截获 spans
        }

    def build_infer_inputs_v1(self, messages):
        self.conv_mode = "v1"
        conv = conv_templates[self.conv_mode].copy()

        prompt = ""

        if "system" in messages:
            prompt = messages["system"]

        if "prompt" in messages:
            prompt = prompt + "\n" + messages["prompt"] if prompt else messages["prompt"]

        images = messages.get("images", None)
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            images = [
                img if isinstance(img, Image.Image)
                else Image.open(img).convert("RGB")
                for img in images
            ]

        prompt = self.move_image_token_to_front(prompt)


        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        prompt_text = conv.get_prompt()

        # 图像处理
        if images is not None:
            image_tensor = process_images(
                images, self.image_processor, self.model.config
            )
            if isinstance(image_tensor, list):
                image_tensor = [
                    img.to(self.model.device, dtype=torch.float16)
                    for img in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(
                    self.model.device, dtype=torch.float16
                )
        else:
            image_tensor = None

        input_ids = tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        attention_mask = torch.ones_like(input_ids).to(self.model.device)
    

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "attention_mask": attention_mask,
        }
    
    def generate_output(self, messages):
        """【你的原版代码完全不动】生成输出（普通推理使用）"""
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
        
        with torch.inference_mode():
            try:
                output_ids = self.model.generate(**generation_kwargs)
            except Exception as e:
                raise
        
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        if generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)].strip()
        
        return generated_text.strip()
    
    def generate_outputs(self, messages_list):
        """【你的原版代码完全不动】批量生成输出"""
        results = []
        for messages in messages_list:
            result = self.generate_output(messages)
            results.append(result)
        return results

    # ==========================================
    # 以下为新增的方法：完全独立，用于可视化注意力
    # ==========================================

    def compute_single_sample_attention_ratio(self, all_step_attentions, modal_token_spans):
        """提取单样本注意力权重比例"""
        generated_len = len(all_step_attentions)
        if generated_len == 0:
            return None, None
            
        first_step = all_step_attentions[0]
        num_layers = len(first_step)

        layer_image_ratio = np.zeros(num_layers)
        layer_clinical_ratio = np.zeros(num_layers)

        spans = modal_token_spans[0] if isinstance(modal_token_spans, list) else modal_token_spans
        if not spans:
            return None, None

        for step_data in all_step_attentions:
            for layer_idx in range(num_layers):
                layer_attn = step_data[layer_idx]  # [1, Heads, q_len, kv_len]
                # 对 head 和 batch 平均 -> [q_len, kv_len]
                attn = layer_attn.mean(dim=1).mean(dim=0)
                
                # 获取当前生成的最后一个 token 对其他 token 的 Attention 分布
                target_attn = attn[-1] 
                total_attn = target_attn.sum().item() + 1e-8

                s, e = spans.get("image") or (0, 0)
                img_w = target_attn[s:e].sum().item() if e > s else 0.0

                cli_w = 0.0
                for key in ["dev", "peri", "phys"]:
                    s, e = spans.get(key) or (0, 0)
                    if e > s:
                        cli_w += target_attn[s:e].sum().item()

                layer_image_ratio[layer_idx] += (img_w / total_attn)
                layer_clinical_ratio[layer_idx] += (cli_w / total_attn)

        layer_image_ratio /= generated_len
        layer_clinical_ratio /= generated_len

        return layer_image_ratio, layer_clinical_ratio

    def plot_averaged_layerwise_attention(self, layer_image_ratio, layer_clinical_ratio, total_samples):
        """画100个样本平均注意力图"""
        mpl.rcParams['font.family'] = 'Times New Roman'
        num_layers = len(layer_image_ratio)

        plt.figure(figsize=(8, 6))
        x = np.arange(num_layers)

        plt.plot(x, layer_image_ratio + 1e-6, linewidth=2, label="Image Tokens")
        plt.plot(x, layer_clinical_ratio + 1e-6, linewidth=2, label="Clinic Tokens")

        plt.xlabel("Decoder Layer Index", fontsize=14)
        plt.ylabel("Attention Ratio", fontsize=14)
        plt.title(f"Average Attention of Different Tokens (N={total_samples})", fontsize=16)

        plt.yscale("log")
        plt.ylim(1e-3, 1)
        plt.yticks([1e-3, 1e-2, 1e-1, 1])

        plt.legend()
        plt.grid(False)
        plt.minorticks_off()

        save_name = f"layerwise_attention_{total_samples}_samples_avg_llavarad.png"
        plt.savefig(save_name, dpi=300)
        print(f"最终平均注意力曲线图已保存至: {save_name}")
        plt.show()

    def generate_and_analyze_single_sample(self, messages, sample_idx):
        """单样本推理解析"""
        inputs = self.process_messages(messages)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        global GLOBAL_CACHE
        GLOBAL_CACHE["spans"] = None
        
        # 挂载 Monkey Patch 拦截 Spans
        if not hasattr(self.model, "_is_patched_for_spans"):
            self.model.prepare_inputs_labels_for_multimodal = types.MethodType(
                custom_prepare_inputs, self.model
            )
            self.model._is_patched_for_spans = True
            
        conv = conv_templates[self.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, inputs['input_ids'])
        
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
            # 必须开启以下两项提取 Attention
            "output_attentions": True,
            "return_dict_in_generate": True,
        }
        
        layer_image_ratio = None
        layer_clinical_ratio = None

        with torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs)
            modal_token_spans = GLOBAL_CACHE.get("spans", None)
            
            if outputs.attentions is not None and modal_token_spans is not None:
                cpu_attentions = []
                for step_attn in outputs.attentions:
                    layers_on_cpu = [layer.detach().cpu() for layer in step_attn]
                    cpu_attentions.append(layers_on_cpu)
                
                layer_image_ratio, layer_clinical_ratio = self.compute_single_sample_attention_ratio(
                    cpu_attentions, modal_token_spans
                )
                del cpu_attentions
            
            # 清理显存
            del outputs
            del inputs
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"[Sample {sample_idx+1}] 完成分析。当前显存占用: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        return layer_image_ratio, layer_clinical_ratio

    def analyze_batch_with_memory_control(self, test_dataset, max_samples=100):
        """【入口】直接运行这个函数画出前 100 个样本的注意力曲线"""
        print(f"开始分析 {max_samples} 条样本并提取注意力权重...")
        
        accumulated_image_ratio = None
        accumulated_clinical_ratio = None
        valid_samples = 0

        for idx in range(min(max_samples, len(test_dataset))):
            print(f"\n{'='*50}")
            print(f"正在处理样本: {idx+1}/{max_samples}")
            
            # 解析你数据集中的字段
            sample = test_dataset[idx]
            messages = {
                "prompt": sample.get("prompt", ""),
                "images": sample.get("image", None),
                "system": "You are a helpful assistant."
            }
            
            img_ratio, cli_ratio = self.generate_and_analyze_single_sample(messages, idx)
            
            if img_ratio is not None and cli_ratio is not None:
                if accumulated_image_ratio is None:
                    accumulated_image_ratio = np.zeros_like(img_ratio)
                    accumulated_clinical_ratio = np.zeros_like(cli_ratio)
                
                accumulated_image_ratio += img_ratio
                accumulated_clinical_ratio += cli_ratio
                valid_samples += 1
            else:
                print(f"[警告] 样本 {idx+1} 丢失了注意力 spans，已跳过。")
            
            gc.collect()
            torch.cuda.empty_cache()

        if valid_samples > 0:
            print(f"\n{'='*50}\n已完成 {valid_samples} 个有效样本，正在生成合并趋势图...")
            avg_image_ratio = accumulated_image_ratio / valid_samples
            avg_clinical_ratio = accumulated_clinical_ratio / valid_samples
            
            self.plot_averaged_layerwise_attention(
                avg_image_ratio, 
                avg_clinical_ratio, 
                valid_samples
            )
        else:
            print("未找到有效带注意力跨度的样本，绘图终止。")


    def generate_output(self, messages, return_attention=False):
        """生成输出，顺便提取注意力权重"""
        # 注意：这里按你提供的代码使用了 build_infer_inputs_v1
        inputs = self.build_infer_inputs_v1(messages)

        global GLOBAL_CACHE
        
        GLOBAL_CACHE["spans"] = None
        if not hasattr(self.model, "_is_patched_for_spans"):
            self.model.prepare_inputs_labels_for_multimodal = types.MethodType(custom_prepare_inputs, self.model)
            self.model._is_patched_for_spans = True
            print(">>> [Init] Monkey Patch Applied Successfully!")
    
        generation_kwargs = {
            "input_ids": inputs['input_ids'], 
            "images": inputs['images'],       
            "do_sample": True if self.temperature > 0 else False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "output_attentions": True, # 必须开启
            "return_dict_in_generate": True,
        }

        layer_image_ratio, layer_clinical_ratio = None, None

        with torch.inference_mode():
            try:
                outputs = self.model.generate(**generation_kwargs)
                modal_token_spans = GLOBAL_CACHE["spans"]
                
                if modal_token_spans is not None:
                    pass # print(">>> [SUCCESS] 成功从全局缓存拿回 last_modal_spans!")
                else:
                    print(">>> [FAIL] 全局缓存为空！可能是 prepare 没有正确执行 prefill 逻辑。")

                if outputs.attentions is not None:
                    cpu_attentions = []
                    for step_attn in outputs.attentions:
                        layers_on_cpu = [layer.detach().cpu() for layer in step_attn]
                        cpu_attentions.append(layers_on_cpu)
                else:
                    cpu_attentions = None
                
                output_ids = outputs.sequences.detach().cpu()

                # ---- 【核心改动：顺便计算单样本 Attention 比例】 ----
                if modal_token_spans is not None and cpu_attentions is not None:
                    layer_image_ratio, layer_clinical_ratio = self.compute_single_sample_attention_ratio(
                        cpu_attentions,
                        modal_token_spans
                    )
                # ----------------------------------------------------

                del outputs 
                torch.cuda.empty_cache() 
                
            except Exception as e:
                print(f"生成失败: {e}")
                torch.cuda.empty_cache() 
                raise
        
        input_token_len = inputs['input_ids'].shape[1]
        
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], 
            skip_special_tokens=True
        )[0]
        
        # 如果需要提取 attention (如在批量画图时)，则返回 tuple
        if return_attention:
            return generated_text.strip(), layer_image_ratio, layer_clinical_ratio
            
        return generated_text.strip()
    
    def generate_outputs(self, messages_list, plot_attention=True):
        """批量生成输出，并顺便累加画出全局注意力曲线"""
        results = []
        
        accumulated_image_ratio = None
        accumulated_clinical_ratio = None
        valid_samples = 0

        for idx, messages in enumerate(messages_list):
            print(f"Processing sample {idx+1}/{len(messages_list)}")
            
            # 使用 return_attention=True 获取返回的比例
            if plot_attention:
                text, img_ratio, cli_ratio = self.generate_output(messages, return_attention=True)
                
                # 累加合法样本的比例
                if img_ratio is not None and cli_ratio is not None:
                    if accumulated_image_ratio is None:
                        accumulated_image_ratio = np.zeros_like(img_ratio)
                        accumulated_clinical_ratio = np.zeros_like(cli_ratio)
                    
                    accumulated_image_ratio += img_ratio
                    accumulated_clinical_ratio += cli_ratio
                    valid_samples += 1
            else:
                # 正常普通生成
                text = self.generate_output(messages, return_attention=False)
                
            results.append(text)
            
            # 每跑完一个强制清理内存
            import gc
            gc.collect()

        # 批量跑完后，顺便画一张总图
        if plot_attention and valid_samples > 0:
            print(f"\n✅ 批量生成结束，正在合并 {valid_samples} 个样本的 Attention 绘制总图...")
            avg_image_ratio = accumulated_image_ratio / valid_samples
            avg_clinical_ratio = accumulated_clinical_ratio / valid_samples
            self.plot_averaged_layerwise_attention(avg_image_ratio, avg_clinical_ratio, valid_samples)

        return results