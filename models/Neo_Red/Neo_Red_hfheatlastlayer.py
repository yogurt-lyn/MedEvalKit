import torch
import os
import sys
import json
from PIL import Image
from peft import PeftModel
sys.path.append("/data01/kjt_data/MLLM/NeoRed")

import torch
import types
GLOBAL_CACHE = {"spans": None}

import matplotlib.pyplot as plt
import seaborn as sns
import torch
# export CUDA_HOME=/usr/local/cuda-11.8
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda-11.8/bin:$PATH

from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


MARKER_ID_PATTERNS = {
    "dev": {
        "start": [29966,3359, 29958],  
        "end":   [829,3359, 29958],
    },
    "peri": {
        "start": [29966, 546, 29875, 29958],
        "end":   [829,546, 29875, 29958],
    },
    "phys": {
        "start": [29966,14017, 29958],
        "end":   [829,14017, 29958],
    },
}
@staticmethod
def find_subseq_ids(seq, pattern):
    """
    seq: List[int] or 1D tensor
    pattern: List[int]
    return: (start_idx, end_idx) or None
    """
    m = len(pattern)
    for i in range(len(seq) - m + 1):
        if seq[i:i + m] == pattern:
            return i, i + m
    return None

@staticmethod
def generate_I_D_mask_from_labels(input_ids, labels):
    if labels is None:
        return None, None
    """
    基于 labels 的有效区域生成 I_mask 和 D_mask，附加 debug 输出
    """
    bsz, seqlen = labels.shape
    I_mask = torch.zeros_like(labels, dtype=torch.bool)
    D_mask = torch.zeros_like(labels, dtype=torch.bool)

    IMAGING_CONCLUSION_IDS = [6751,15997, 29901]
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

        # I_mask 覆盖区间：marker 后到 D_marker 前
        # im_content_start = start-4 + im_start + len(IMAGING_CONCLUSION_IDS)
        im_content_start = start-3 + im_start + len(IMAGING_CONCLUSION_IDS)
        im_content_end   = start + d_start -1 
    
        # D_mask 覆盖区间：D_marker 后到 labels 末尾
        d_content_start = start-6 + d_start + len(DISEASE_DIAGNOSIS_IDS)
        d_content_end   = end

        # 填充 mask
        I_mask[b, im_content_start:im_content_end] = True
        D_mask[b, d_content_start:d_content_end] = True
        
    I_mask = [I_mask[b] for b in range(bsz)]
    D_mask = [D_mask[b] for b in range(bsz)]
    return I_mask, D_mask

def custom_prepare_inputs(self, input_ids, attention_mask, past_key_values, labels, images):
    """
    独立于类的自定义预处理函数，用于 Monkey Patch
    """
    vision_tower = self.get_vision_tower()
    
    # ------------------------------------------------------------------
    # 1. Early Exit (快速退出逻辑 - 解码阶段)
    # ------------------------------------------------------------------
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
             attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
        
        # [修改点 1] 强制返回 8 个值，补齐最后 3 个为 None
        return input_ids, attention_mask, past_key_values, None, labels, None, None, None

    # ------------------------------------------------------------------
    # 2. Main Logic (主逻辑 - 预填充阶段)
    # ------------------------------------------------------------------
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
        # ... (以下是你提供的原始循环逻辑，原封不动) ...
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
            assert cur_labels.shape == cur_input_ids.shape
            
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

        # ... (Spans 计算逻辑) ...
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

        # Image Offset
        num_images = (cur_input_ids_raw == IMAGE_TOKEN_INDEX).sum().item()
        total_offset = 0
        if num_images > 0:
            img_len = image_features[0].shape[0]
            total_offset = num_images * (img_len - 1)

        if total_offset > 0 and I_mask is not None and D_mask is not None:
            cur_I_mask = I_mask[batch_idx]
            cur_D_mask = D_mask[batch_idx]
            
            if total_offset > 0:
                new_len = cur_I_mask.shape[0] + total_offset
                shifted_I = torch.zeros(new_len, dtype=cur_I_mask.dtype, device=cur_I_mask.device)
                shifted_D = torch.zeros(new_len, dtype=cur_D_mask.dtype, device=cur_D_mask.device)
                shifted_I[total_offset:] = cur_I_mask
                shifted_D[total_offset:] = cur_D_mask
                I_mask[batch_idx] = shifted_I
                D_mask[batch_idx] = shifted_D

        # Image Spans
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

        # Text Markers
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

    # ------------------------------------------------------------------
    # [修改点 2] 存入全局缓存 (仅在 input_ids 长度 > 1 即预填充时)
    # ------------------------------------------------------------------
    if input_ids.shape[1] > 1:
        global GLOBAL_CACHE
        GLOBAL_CACHE["spans"] = modal_token_spans
        print(">>> [Custom Prepare] Spans computed and saved to GLOBAL_CACHE.")

    # ... (对齐逻辑保持不变) ...
    if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
        max_len = max(x.shape[0] for x in new_input_embeds)
        new_input_embeds_align = []
        for cur_new_embed in new_input_embeds:
            cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
            new_input_embeds_align.append(cur_new_embed)
        new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

        if labels is not None:
            new_labels_align = []
            _new_labels = new_labels
            for cur_new_label in new_labels:
                cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                new_labels_align.append(cur_new_label)
            new_labels = torch.stack(new_labels_align, dim=0)

        if attention_mask is not None:
            new_attention_mask = []
            for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                new_attention_mask.append(cur_new_attention_mask)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            assert attention_mask.shape == new_labels.shape

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
            new_labels  = torch.stack(new_labels, dim=0)
        if I_mask is not None and D_mask is not None:
            I_mask = torch.stack(I_mask, dim=0)
            D_mask = torch.stack(D_mask, dim=0)
        if attention_mask is not None:
            new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
            assert attention_mask.shape == new_input_embeds.shape[:2]

    # [修改点 3] 统一返回 8 个值, modal_token_spans, I_mask, D_mask
    return None, attention_mask, past_key_values, new_input_embeds, new_labels, modal_token_spans, I_mask, D_mask

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


    # 假设这个函数是在类外部定义的，或者你需要给它加上 @staticmethod
    # 如果是在类内部，请确保第一个参数是 self，调用时用 self.plot...

    # def plot_token_modality_attention(self,outputs, test_modal_token_spans, tokenizer, batch_idx=0):
    #     """
    #     可视化注意力热图
    #     """
    #     # --- 1. 获取 Attention 数据 ---
    #     all_step_attentions = outputs.attentions 
        
    #     if all_step_attentions is None:
    #         raise ValueError("outputs 中没有 attentions，请检查 generate 时是否设置了 output_attentions=True")

    #     # --- 2. 获取生成的 Token ---
    #     # outputs.sequences 包含 [Input_Prompt + Generated_Tokens]
    #     full_seq_ids = outputs.sequences[batch_idx]
        
    #     # 我们只关心新生成的 token
    #     generated_len = len(all_step_attentions)
    #     gen_ids = full_seq_ids[-generated_len:] 
        
    #     # 解码 (使用 tokenizer 而不是 processor)
    #     gen_words = [tokenizer.decode([mid], skip_special_tokens=True) for mid in gen_ids]

    #     # --- 3. 准备绘图 ---
    #     target_keys = ["image", "dev", "peri", "phys"]
    #     spans = test_modal_token_spans[batch_idx]
        
    #     # matrix: [模态数, 生成词数]
    #     attn_matrix = torch.zeros(len(target_keys), generated_len)

    #     # --- 4. 填充矩阵 ---
    #     for step_idx, step_attn in enumerate(all_step_attentions):
    #         # 取最后一层，对 Head 取平均 -> [1, Past_Len]
    #         layer_attn_avg = step_attn[-1][batch_idx].mean(dim=0) 
            
    #         for k_idx, key in enumerate(target_keys):
    #             s, e = spans.get(key, (0, 0))
    #             if e > s:
    #                 # 累加对该区域的注意力
    #                 weight = layer_attn_avg[0, s:e].sum().item()
    #                 attn_matrix[k_idx, step_idx] = weight
    
    def plot_token_modality_attention(self, full_seq_ids, all_step_attentions, test_modal_token_spans, tokenizer, batch_idx=0):
        """
        full_seq_ids: [Batch, Seq_Len] (CPU Tensor)
        all_step_attentions: List of [Batch, Heads, 1, Seq_Len] (CPU Tensors, 仅包含最后一层)
        """
        
        # --- 1. 获取生成的 Token ---
        generated_len = len(all_step_attentions)
        
        # 切片取新生成的 token
        gen_ids = full_seq_ids[0][-generated_len:] # 注意 batch 维度
        
        # 解码
        gen_words = [tokenizer.decode([mid.item()], skip_special_tokens=True) for mid in gen_ids]

        # --- 2. 准备绘图 ---
        target_keys = ["image", "dev", "peri", "phys"]
        spans = test_modal_token_spans[batch_idx]
        
        attn_matrix = torch.zeros(len(target_keys), generated_len)

        # --- 3. 填充矩阵 ---
        for step_idx, layer_attn in enumerate(all_step_attentions):
            # layer_attn 已经是最后一层了，不需要再 [-1]
            # shape: [Batch, Heads, 1, Past_Len]
            
            # 对 Head 取平均
            layer_attn_avg = layer_attn[batch_idx].mean(dim=0) 
            
            for k_idx, key in enumerate(target_keys):
                s, e = spans.get(key, (0, 0))
                if e > s:
                    weight = layer_attn_avg[0, s:e].sum().item()
                    attn_matrix[k_idx, step_idx] = weight

        # --- 5. 绘图 ---
        plt.figure(figsize=(20, 8))
        sns.heatmap(
            attn_matrix.numpy(), 
            xticklabels=gen_words, 
            yticklabels=target_keys, 
            cmap="coolwarm",
            annot=False, 
            fmt=".2f"
        )
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=12)
        plt.title("Attention Heatmap", fontsize=14)
        plt.xlabel("Generated Tokens")
        plt.ylabel("Input Modality")
        plt.tight_layout()

        plt.show()

        plt.tight_layout()
    
        # --- 新增保存代码 ---
        # 1. 定义文件名 (建议加上 batch_idx 甚至时间戳，防止覆盖)
        save_name = f"attention_heatmap_batch_{batch_idx}.png"
        # 或者保存为矢量图用于论文 (PDF格式放大不模糊)
        # save_name = f"attention_heatmap_batch_{batch_idx}.pdf"
        
        # 2. 保存图片
        # dpi=300: 论文标准的高清分辨率
        # bbox_inches='tight': 防止边缘的标签被切掉
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_name}")
        
        # 3. 显示图片 (这行必须在 savefig 之后)
        plt.show()

        # 显存清理
        del all_step_attentions
        torch.cuda.empty_cache()

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

    # def generate_output(self, messages):
    #     """生成输出"""
    #     inputs = self.build_infer_inputs_v1(messages)
    #     Images=inputs['images']
    #     # print(Images.shape)
    #     with torch.no_grad():
    #         (
    #             _, # input_ids 被替换为 embeds 了
    #             attention_mask, 
    #             past_key_values, 
    #             inputs_embeds, 
    #             _, 
    #             test_modal_token_spans, 
    #             I_mask, 
    #             D_mask
    #         ) = self.model.prepare_inputs_labels_for_multimodal(
    #             inputs['input_ids'], 
    #             inputs['attention_mask'], 
    #             None, # past_key_values
    #             None, # labels
    #             inputs['images']
    #         )
        
    #    #######一堆调试代码#########
    #     # 生成参数
    #     generation_kwargs = {
    #         "input_ids": inputs['input_ids'],
    #         "images": inputs['images'],
    #         "do_sample": True if self.temperature > 0 else False,
    #         "temperature": self.temperature,
    #         "top_p": self.top_p,
    #         "repetition_penalty": self.repetition_penalty,
    #         "max_new_tokens": self.max_new_tokens,  # 使用原始值，不调整
    #         "use_cache": True,
    #         "output_attentions": True,
    #         "return_dict_in_generate": True,
    #         # "stopping_criteria": [stopping_criteria],
    #     }


    #     # 生成输出
    #     with torch.inference_mode():
    #         try:
    #             # output_ids = self.model.generate(**generation_kwargs)
    #             outputs = self.model.generate(**generation_kwargs)
    #             output_ids = outputs.sequences
    #             all_attentions = outputs.attentions
    #             self.plot_token_modality_attention(outputs, test_modal_token_spans, self.tokenizer, batch_idx=0)
    #         except Exception as e:
    #             print(f"生成失败: {e}")
    #             # if total_input_length > max_context_len:
    #             #     return f"错误: 输入序列长度({total_input_length})超过模型最大上下文长度({max_context_len})。请减少图片数量或缩短prompt。"
    #             raise
        
    #     # 解码输出（跳过输入token）
    #     generated_text = self.tokenizer.batch_decode(
    #         output_ids[:, inputs['input_ids'].shape[1]:], 
    #         skip_special_tokens=True
    #     )[0]
        
    #     return generated_text.strip()
    def generate_output(self, messages):
        """生成输出"""
        inputs = self.build_infer_inputs_v1(messages)

        global GLOBAL_CACHE
        
        # 为了安全，每次生成前清空一下缓存
        GLOBAL_CACHE["spans"] = None
        if not hasattr(self.model, "_is_patched_for_spans"):
            self.model.prepare_inputs_labels_for_multimodal = types.MethodType(custom_prepare_inputs, self.model)
            self.model._is_patched_for_spans = True
            print(">>> [Init] Monkey Patch Applied Successfully!")
    
        generation_kwargs = {
            # "inputs_embeds": inputs_embeds, # 传入 Embeds
            # "attention_mask": attention_mask_expanded, # 传入扩展后的 Mask
            "input_ids": inputs['input_ids'], # 不要传这个，防止冲突
            "images": inputs['images'],       # 不要传这个，embeds里已经有了
            "do_sample": True if self.temperature > 0 else False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "output_attentions": True, # 必须开启
            "return_dict_in_generate": True,
        }

    #     # 生成输出
    #     with torch.inference_mode():
    #         try:
    #             outputs = self.model.generate(**generation_kwargs)
                
    #             if hasattr(self.model, "last_modal_spans"):
    #                 modal_token_spans = self.model.last_modal_spans
    #             else:
    #                 print("Warning: last_modal_spans not found on model!")
    #                 modal_token_spans = None

    #             # 3. 调用绘图
    #             if modal_token_spans is not None:
    #                 # 注意：第一个参数传 outputs (ModelOutput对象)，不要传 string
    #                 self.plot_token_modality_attention(outputs, modal_token_spans, self.tokenizer, batch_idx=0)
                
    #         except Exception as e:
    #             print(f"生成失败: {e}")
    #             raise
        with torch.inference_mode():
            try:
                # 1. 生成 (此时显存压力最大)
                outputs = self.model.generate(**generation_kwargs)
                
                # =======================================================
                # 【显存优化核心】: 不要把整个 outputs 传给画图函数！
                # =======================================================
                
                modal_token_spans = GLOBAL_CACHE["spans"]
                
                if modal_token_spans is not None:
                    print(">>> [SUCCESS] 成功从全局缓存拿回 last_modal_spans!")
                else:
                    print(">>> [FAIL] 全局缓存为空！可能是 prepare 没有正确执行 prefill 逻辑。")

                # 3. 【关键步骤】提取 Attention 并立即转到 CPU
                # 我们只需要: 每一跳(step) -> 最后一层(-1) -> 的注意力
                # 这样做会把几 GB 的 GPU Tensor 变成几 MB 的 CPU Tensor
                if outputs.attentions is not None:
                    cpu_attentions = []
                    for step_attn in outputs.attentions:
                        # step_attn 是一个 tuple(所有层)，我们只取最后一层 [-1]
                        # .cpu() 将其从显存挪走
                        # .detach() 切断梯度（虽然inference mode本来就没有）
                        last_layer_on_cpu = step_attn[-1].detach().cpu()
                        cpu_attentions.append(last_layer_on_cpu)
                else:
                    cpu_attentions = None
                
                # 4. 获取 Token IDs (也转到 CPU)
                output_ids = outputs.sequences.detach().cpu()

                # 5. 【立即删除】销毁 GPU 上的巨型对象
                del outputs 
                torch.cuda.empty_cache() # 强制回收显存

                # 6. 调用绘图 (注意：现在传的是 cpu_attentions)
                # 你需要稍微修改一下 plot 函数来适应这个 list 结构
                if modal_token_spans is not None and cpu_attentions is not None:
                    print("plot_token_modality_attention Get!")
                    self.plot_token_modality_attention(
                        output_ids,      # 传 IDs
                        cpu_attentions,  # 传 CPU 上的 list
                        modal_token_spans, 
                        self.tokenizer, 
                        batch_idx=0
                    )
                
            except Exception as e:
                print(f"生成失败: {e}")
                # 即使失败也要清理显存
                torch.cuda.empty_cache() 
                raise
        
        input_token_len = inputs['input_ids'].shape[1]
        # output_ids = outputs.sequences
        
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], 
            skip_special_tokens=True
        )[0]
        
        return generated_text.strip()
    
    def generate_outputs(self, messages_list):
        """批量生成输出"""
        results = []
        for messages in messages_list:
            result = self.generate_output(messages)
            results.append(result)
        return results
    