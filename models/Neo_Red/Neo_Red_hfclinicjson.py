import torch
import os
import sys
import json
from PIL import Image
from peft import PeftModel
sys.path.append("/data01/kjt_data/MLLM/NeoRed")
import math
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
        model_base = os.path.join("/data01/kjt_data/MLLM/LLaVARadWeight", "vicuna-7b-v1.5")
        
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
  

    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    def plot_layerwise_modality_attention_line(
        self,
        full_seq_ids,
        all_step_attentions,
        test_modal_token_spans,
        tokenizer,
        batch_idx=0):

        import torch
        import numpy as np
        import matplotlib.pyplot as plt

        target_keys = ["image", "dev", "peri", "phys"]
        spans = test_modal_token_spans[batch_idx]

        generated_len = len(all_step_attentions)

        # -------- 推断层数 --------
        first_step = all_step_attentions[0]
        if isinstance(first_step, (list, tuple)):
            num_layers = len(first_step)
        else:
            num_layers = first_step.shape[1]

        print(f"Layers: {num_layers}, Steps: {generated_len}")

        # -------- 存储矩阵 [Layers, Modalities] --------
        # layer_modality_ratio = torch.zeros(num_layers, len(target_keys))
        layer_image_ratio = torch.zeros(num_layers)
        layer_clinical_ratio = torch.zeros(num_layers)

        for step_data in all_step_attentions:

            for layer_idx in range(num_layers):

                if isinstance(step_data, (list, tuple)):
                    layer_attn = step_data[layer_idx]
                else:
                    layer_attn = step_data[:, layer_idx, ...]

                attn = layer_attn[batch_idx].mean(dim=0)  # [1, Seq]

                total_attn = attn[0].sum().item() + 1e-8

                # -------- image --------
                s, e = spans.get("image", (0, 0))
                if e > s:
                    image_weight = attn[0, s:e].sum().item()
                    layer_image_ratio[layer_idx] += image_weight / total_attn

                # -------- clinical text (dev+peri+phys) --------
                clinical_weight = 0.0
                for key in ["dev", "peri", "phys"]:
                    s, e = spans.get(key, (0, 0))
                    if e > s:
                        clinical_weight += attn[0, s:e].sum().item()

                layer_clinical_ratio[layer_idx] += clinical_weight / total_attn


        layer_image_ratio /= generated_len
        layer_clinical_ratio /= generated_len

        layer_image_ratio = layer_image_ratio.numpy()
        layer_clinical_ratio = layer_clinical_ratio.numpy()

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mpl.rcParams['font.family'] = 'Times New Roman'

        plt.figure(figsize=(8, 6))
        x = np.arange(num_layers)

        plt.plot(
            x,
            layer_image_ratio + 1e-6,
            label="Image Tokens",
            linewidth=2
        )

        plt.plot(
            x,
            layer_clinical_ratio + 1e-6,
            label="Clinic Tokens",
            linewidth=2
        )

        plt.xlabel("Decoder Layer Index", fontsize=14)
        plt.ylabel("Attention Ratio", fontsize=14)
        plt.title("Attention of Different Tokens", fontsize=16)

        plt.yscale("log")
        plt.ylim(1e-3, 1)
        plt.yticks([1e-3, 1e-2, 1e-1, 1])

        plt.legend()
        # plt.grid(alpha=0.3, which="both")
        plt.grid(False)
        plt.minorticks_off()

        save_name = f"layerwise_attention_image_vs_clinical_batch_{batch_idx}_Neo.png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved to {save_name}")

        plt.show()

    def plot_all_layers_token_modality_attention(self, full_seq_ids, all_step_attentions, test_modal_token_spans, tokenizer, batch_idx=0):
        """
        full_seq_ids: [Batch, Seq_Len]
        all_step_attentions: List (Steps) of List (Layers) of Tensor [Batch, Heads, 1, Seq_Len]
                            或者 List (Steps) of Tensor [Batch, Layers, Heads, 1, Seq_Len]
                            **关键：必须包含所有层的 Attention**
        """
        
        # --- 1. 获取生成的 Token ---
        generated_len = len(all_step_attentions)
        gen_ids = full_seq_ids[0][-generated_len:] 
        gen_words = [tokenizer.decode([mid.item()], skip_special_tokens=True) for mid in gen_ids]

        # --- 2. 准备数据容器 ---
        target_keys = ["image", "dev", "peri", "phys"]
        spans = test_modal_token_spans[batch_idx]
        
        # 自动推断层数 (假设 all_step_attentions[0] 是一个包含所有层 Tensor 的列表或堆叠后的 Tensor)
        first_step_data = all_step_attentions[0]
        if isinstance(first_step_data, (list, tuple)):
            num_layers = len(first_step_data)
        else:
            # 如果是 Tensor [Batch, Layers, Heads, ...]
            num_layers = first_step_data.shape[1]

        print(f"检测到模型层数: {num_layers}, 生成步数: {generated_len}")

        # 创建一个 3D 矩阵: [Layers, Modality_Keys, Generated_Steps]
        # 我们用列表存储每一层的矩阵
        all_layers_matrices = [torch.zeros(len(target_keys), generated_len) for _ in range(num_layers)]

        # --- 3. 填充矩阵 (遍历每一步、每一层) ---
        for step_idx, step_layers_data in enumerate(all_step_attentions):
            # step_layers_data: 包含了这一步所有层的信息
            
            for layer_idx in range(num_layers):
                # 获取特定层的 attention tensor
                if isinstance(step_layers_data, (list, tuple)):
                    layer_attn = step_layers_data[layer_idx] # [Batch, Heads, 1, Past_Len]
                else:
                    layer_attn = step_layers_data[:, layer_idx, ...] # [Batch, Heads, 1, Past_Len]

                # 对 Head 取平均
                # layer_attn[batch_idx] shape: [Heads, 1, Past_Len] -> mean(0) -> [1, Past_Len]
                layer_attn_avg = layer_attn[batch_idx].mean(dim=0) 
                
                # 提取特定模态的权重
                for k_idx, key in enumerate(target_keys):
                    s, e = spans.get(key, (0, 0))
                    if e > s:
                        # 注意：这就要求 layer_attn 覆盖了原图输入的长度
                        weight = layer_attn_avg[0, s:e].sum().item()
                        all_layers_matrices[layer_idx][k_idx, step_idx] = weight

        # --- 4. 绘图 (网格布局) ---
        # 计算网格行列，例如 32层 -> 4行8列 或 6行6列
        cols = 4 
        rows = math.ceil(num_layers / cols)
        
        # 设置画布大小，行数越多图片越高
        fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows), constrained_layout=True)
        axes_flat = axes.flatten()

        for layer_idx in range(num_layers):
            ax = axes_flat[layer_idx]
            matrix = all_layers_matrices[layer_idx].numpy()
            
            sns.heatmap(
                matrix, 
                xticklabels=gen_words if layer_idx >= (rows-1)*cols else [], # 只在最后几行显示文字，避免拥挤
                yticklabels=target_keys, 
                cmap="YlGnBu",
                annot=False, 
                cbar=True, # 可以选择关掉 cbar 节省空间
                ax=ax
            )
            
            ax.set_title(f"Layer {layer_idx}")
            # 旋转 x 轴标签
            if layer_idx >= (rows-1)*cols:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

        # 隐藏多余的子图
        for i in range(num_layers, len(axes_flat)):
            axes_flat[i].axis('off')

        # --- 5. 保存与显示 ---
        save_name = f"all_layers_attention_batch_{batch_idx}__Neo.png"
        plt.savefig(save_name, dpi=150) # 层数多时 dpi 稍微低点防止文件过大
        print(f"全层注意力热力图已保存至: {save_name}")
        
        plt.show()

        # 清理
        del all_layers_matrices
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

    # def extract_attention_ratios(self, all_step_attentions, test_modal_token_spans, batch_idx=0):
    #     """
    #     核心精简版：只计算并提取当前样本的 Image 和 Clinical 的 Attention 比例
    #     """
    #     spans = test_modal_token_spans[batch_idx] if test_modal_token_spans else {}
    #     generated_len = len(all_step_attentions)
        
    #     if generated_len == 0:
    #         return [], []

    #     # 推断层数
    #     first_step = all_step_attentions[0]
    #     if isinstance(first_step, (list, tuple)):
    #         num_layers = len(first_step)
    #     else:
    #         num_layers = first_step.shape[1]

    #     # 用普通列表存，方便最后存 JSON
    #     layer_image_ratio = [0.0] * num_layers
    #     layer_clinical_ratio = [0.0] * num_layers

    #     if not spans:
    #         return layer_image_ratio, layer_clinical_ratio

    #     for step_data in all_step_attentions:
    #         for layer_idx in range(num_layers):
    #             if isinstance(step_data, (list, tuple)):
    #                 layer_attn = step_data[layer_idx]
    #             else:
    #                 layer_attn = step_data[:, layer_idx, ...]

    #             # 对 heads 平均
    #             attn = layer_attn[batch_idx].mean(dim=0)  # [1, Seq]

    #             # 获取当前生成的 Token 的 Attention 分布
    #             # prefill 时有多个 query，取最后一个；decode 时只有 1 个 query，直接取
    #             target_attn = attn[-1] if attn.shape[0] > 1 else attn[0]
    #             total_attn = target_attn.sum().item() + 1e-8

    #             # -------- 1. Image --------
    #             s, e = spans.get("image") or (0, 0)
    #             if e > s:
    #                 layer_image_ratio[layer_idx] += target_attn[s:e].sum().item() / total_attn

    #             # -------- 2. Clinical Text --------
    #             clinical_weight = 0.0
    #             for key in ["dev", "peri", "phys"]:
    #                 s_c, e_c = spans.get(key) or (0, 0)
    #                 if e_c > s_c:
    #                     clinical_weight += target_attn[s_c:e_c].sum().item()
                
    #             layer_clinical_ratio[layer_idx] += clinical_weight / total_attn

    #     # 求时间步（生成长度）上的平均
    #     layer_image_ratio = [val / generated_len for val in layer_image_ratio]
    #     layer_clinical_ratio = [val / generated_len for val in layer_clinical_ratio]

    #     return layer_image_ratio, layer_clinical_ratio

    # def extract_attention_ratios(self, all_step_attentions, test_modal_token_spans, gen_ids, batch_idx=0):
    #     """
    #     修改版：只计算生成 token 中属于 D_mask 区域（疾病诊断部分）的 Attention 比例
    #     """
    #     spans = test_modal_token_spans[batch_idx] if test_modal_token_spans else {}
    #     generated_len = len(all_step_attentions)
        
    #     if generated_len == 0:
    #         return [], []

    #     # ---------------------------------------------------------
    #     # 【新增逻辑】：在生成的 Token 中动态寻找 D_mask 的起始位置
    #     # ---------------------------------------------------------
    #     gen_ids_list = gen_ids.tolist()
    #     DISEASE_DIAGNOSIS_IDS  = [559, 24876, 19263, 29901]
        
    #     d_start_step = 0 # 默认从0开始（以防万一没找到匹配的 marker）
    #     for i in range(len(gen_ids_list) - len(DISEASE_DIAGNOSIS_IDS) + 1):
    #         if gen_ids_list[i:i+len(DISEASE_DIAGNOSIS_IDS)] == DISEASE_DIAGNOSIS_IDS:
    #             # 找到 marker 后，D_mask 的实际内容从 marker 之后开始
    #             d_start_step = i + len(DISEASE_DIAGNOSIS_IDS)
    #             break
                
    #     # 计算实际属于 D_mask 区域的生成步数
    #     d_mask_len = generated_len - d_start_step
    #     if d_mask_len <= 0:
    #         d_mask_len = 1  # 避免除以 0

    #     # 推断层数
    #     first_step = all_step_attentions[0]
    #     if isinstance(first_step, (list, tuple)):
    #         num_layers = len(first_step)
    #     else:
    #         num_layers = first_step.shape[1]

    #     layer_image_ratio = [0.0] * num_layers
    #     layer_clinical_ratio = [0.0] * num_layers

    #     if not spans:
    #         return layer_image_ratio, layer_clinical_ratio

    #     for step_idx, step_data in enumerate(all_step_attentions):
    #         # ---------------------------------------------------------
    #         # 【关键修改】：如果当前生成的 token 不在 D_mask 范围内，直接跳过！
    #         # ---------------------------------------------------------
    #         if step_idx < d_start_step:
    #             continue

    #         for layer_idx in range(num_layers):
    #             if isinstance(step_data, (list, tuple)):
    #                 layer_attn = step_data[layer_idx]
    #             else:
    #                 layer_attn = step_data[:, layer_idx, ...]

    #             # 对 heads 平均
    #             attn = layer_attn[batch_idx].mean(dim=0)  # [1, Seq]

    #             target_attn = attn[-1] if attn.shape[0] > 1 else attn[0]
    #             total_attn = target_attn.sum().item() + 1e-8

    #             # -------- 1. Image --------
    #             s, e = spans.get("image") or (0, 0)
    #             if e > s:
    #                 layer_image_ratio[layer_idx] += target_attn[s:e].sum().item() / total_attn

    #             # -------- 2. Clinical Text --------
    #             clinical_weight = 0.0
    #             for key in ["dev", "peri", "phys"]:
    #                 s_c, e_c = spans.get(key) or (0, 0)
    #                 if e_c > s_c:
    #                     clinical_weight += target_attn[s_c:e_c].sum().item()
                
    #             layer_clinical_ratio[layer_idx] += clinical_weight / total_attn

    #     # ---------------------------------------------------------
    #     # 【关键修改】：求平均时，除以 D_mask_len 而不是总 generated_len
    #     # ---------------------------------------------------------
    #     layer_image_ratio = [val / d_mask_len for val in layer_image_ratio]
    #     layer_clinical_ratio = [val / d_mask_len for val in layer_clinical_ratio]

    #     return layer_image_ratio, layer_clinical_ratio

    def extract_attention_ratios(self, all_step_attentions, test_modal_token_spans, batch_idx=0):
        """
        更新版：计算 Image, dev, peri, phys 以及总的 Clinical 比例
        """
        spans = test_modal_token_spans[batch_idx] if test_modal_token_spans else {}
        generated_len = len(all_step_attentions)
        
        if generated_len == 0:
            return {}

        first_step = all_step_attentions[0]
        num_layers = len(first_step) if isinstance(first_step, (list, tuple)) else first_step.shape[1]

        # 定义我们要提取的所有细分 key
        sub_clinical_keys = ["dev", "peri", "phys"]
        
        # 初始化结果字典
        # image: 图像
        # clinical_total: 三种临床文本的总和 (即你原来的 cli_ratio)
        res = {
            "image": [0.0] * num_layers,
            "clinical_total": [0.0] * num_layers,
            "dev": [0.0] * num_layers,
            "peri": [0.0] * num_layers,
            "phys": [0.0] * num_layers
        }

        if not spans:
            return res

        for step_data in all_step_attentions:
            for layer_idx in range(num_layers):
                if isinstance(step_data, (list, tuple)):
                    layer_attn = step_data[layer_idx]
                else:
                    layer_attn = step_data[:, layer_idx, ...]

                attn = layer_attn[batch_idx].mean(dim=0)
                target_attn = attn[-1] if attn.shape[0] > 1 else attn[0]
                total_attn = target_attn.sum().item() + 1e-8

                # 1. 计算图像
                s_i, e_i = spans.get("image") or (0, 0)
                if e_i > s_i:
                    res["image"][layer_idx] += target_attn[s_i:e_i].sum().item() / total_attn

                # 2. 计算细分临床文本并累加到总额
                clinical_sum_this_layer = 0.0
                for key in sub_clinical_keys:
                    s, e = spans.get(key) or (0, 0)
                    if e > s:
                        weight = target_attn[s:e].sum().item() / total_attn
                        res[key][layer_idx] += weight
                        clinical_sum_this_layer += weight
                
                res["clinical_total"][layer_idx] += clinical_sum_this_layer

        # 求时间步平均
        for k in res.keys():
            res[k] = [val / generated_len for val in res[k]]

        return res
    
    def generate_output(self, messages, extract_attn=True):
        """生成输出，并顺便提取注意力（包含细分临床文本）"""
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
            "output_attentions": extract_attn,
            "return_dict_in_generate": True,
        }

        # 初始化存储字典
        ratios = {} 

        with torch.inference_mode():
            try:
                outputs = self.model.generate(**generation_kwargs)
                modal_token_spans = GLOBAL_CACHE["spans"]
                
                output_ids = outputs.sequences.detach().cpu()
                input_token_len = inputs['input_ids'].shape[1]
                
                if extract_attn and outputs.attentions is not None:
                    # 转换到 CPU 避免显存堆积
                    cpu_attentions = [[layer.detach().cpu() for layer in step_attn] for step_attn in outputs.attentions]

                    if modal_token_spans is not None:
                        # 【核心修改】：调用更新后的提取函数
                        # 假设 extract_attention_ratios 返回一个包含所有 key 的字典
                        ratios = self.extract_attention_ratios(
                            cpu_attentions, 
                            modal_token_spans, 
                            batch_idx=0
                        )
                    
                    del cpu_attentions # 显式删除大对象
                
                # 清理
                del outputs
                torch.cuda.empty_cache() 
                
            except Exception as e:
                print(f"生成失败: {e}")
                torch.cuda.empty_cache() 
                raise
        
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], 
            skip_special_tokens=True
        )[0]
        
        # 【修改点】：返回 text 以及包含所有比例的字典
        # 这样在 generate_outputs 里可以通过 ratios.get("image") 等拿到所有数据
        return generated_text.strip(), ratios

    # def generate_output(self, messages, extract_attn=True):
    #     """生成输出，并顺便提取注意力"""
    #     inputs = self.build_infer_inputs_v1(messages)

    #     global GLOBAL_CACHE
    #     GLOBAL_CACHE["spans"] = None
        
    #     if not hasattr(self.model, "_is_patched_for_spans"):
    #         self.model.prepare_inputs_labels_for_multimodal = types.MethodType(custom_prepare_inputs, self.model)
    #         self.model._is_patched_for_spans = True
    #         print(">>> [Init] Monkey Patch Applied Successfully!")
    
    #     generation_kwargs = {
    #         "input_ids": inputs['input_ids'], 
    #         "images": inputs['images'],       
    #         "do_sample": True if self.temperature > 0 else False,
    #         "temperature": self.temperature,
    #         "top_p": self.top_p,
    #         "repetition_penalty": self.repetition_penalty,
    #         "max_new_tokens": self.max_new_tokens,
    #         "use_cache": True,
    #         "output_attentions": extract_attn, # 如果要提取，必须为 True
    #         "return_dict_in_generate": True,
    #     }

    #     img_ratio, cli_ratio = [], []
    #     modality_ratios = {} # 存储结果

    #     with torch.inference_mode():
    #         try:
    #             outputs = self.model.generate(**generation_kwargs)
    #             modal_token_spans = GLOBAL_CACHE["spans"]
                
    #             # 提前拿到 output_ids 和生成的 gen_ids
    #             output_ids = outputs.sequences.detach().cpu()
    #             input_token_len = inputs['input_ids'].shape[1]
    #             gen_ids = output_ids[0, input_token_len:] # 获取纯生成的 token
                
    #             if extract_attn and outputs.attentions is not None:
    #                 cpu_attentions = []
    #                 for step_attn in outputs.attentions:
    #                     layers_on_cpu = [layer.detach().cpu() for layer in step_attn]
    #                     cpu_attentions.append(layers_on_cpu)

    #                 # 提取当前样本的 Attention
    #                 if modal_token_spans is not None:
    #                     img_ratio, cli_ratio = self.extract_attention_ratios(
    #                         cpu_attentions, 
    #                         modal_token_spans, 
    #                         # gen_ids,           # <--- 这里把 gen_ids 传进去
    #                         batch_idx=0
    #                     )
                
    #             output_ids = outputs.sequences.detach().cpu()
                
    #             # 清理显存
    #             del outputs
    #             if extract_attn and 'cpu_attentions' in locals():
    #                 del cpu_attentions
    #             torch.cuda.empty_cache() 
                
    #         except Exception as e:
    #             print(f"生成失败: {e}")
    #             torch.cuda.empty_cache() 
    #             raise
        
    #     input_token_len = inputs['input_ids'].shape[1]
    #     generated_text = self.tokenizer.batch_decode(
    #         output_ids[:, input_token_len:], 
    #         skip_special_tokens=True
    #     )[0]
        
    #     # 将文本和提取到的比例一起返回
    #     return generated_text.strip(), img_ratio, cli_ratio

    def generate_outputs(self, messages_list, save_attn_path="attention_resultsLLaVARADT.json"):
        results = []
        attention_data = [] 
        
        for idx, messages in enumerate(messages_list):
            print(f"Processing sample {idx+1}/{len(messages_list)}")
            
            # 接收返回的字典
            text, ratios = self.generate_output(messages, extract_attn=True)
            results.append(text)
            
            # 写入 JSON，同时保留你原来的 key 名
            attention_data.append({
                "sample_id": idx,
                "generated_text": text,
                # 原始聚合指标
                "layer_image_attention_ratio": ratios.get("image", []),
                "layer_clinical_attention_ratio": ratios.get("clinical_total", []),
                # 新增细分指标
                "layer_dev_attention": ratios.get("dev", []),
                "layer_peri_attention": ratios.get("peri", []),
                "layer_phys_attention": ratios.get("phys", [])
            })
            
            import gc
            gc.collect()

        with open(save_attn_path, "w", encoding="utf-8") as f:
            json.dump(attention_data, f, indent=4, ensure_ascii=False)
        
        return results
    
    # def generate_outputs(self, messages_list, save_attn_path="attention_results_llavarad.json"):
    #     """批量生成输出，并自动将每个样本的 Attention 存为 JSON"""
    #     results = []
    #     attention_data = [] # 用于存储所有样本的数据
        
    #     for idx, messages in enumerate(messages_list):
    #         print(f"Processing sample {idx+1}/{len(messages_list)}")
            
    #         # 生成文本并提取注意力
    #         text, ratios = self.generate_output(messages, extract_attn=True)
    #         results.append(text)
            
    #         # 存入字典
    #         attention_data.append({
    #             "sample_id": idx,
    #             "generated_text": text,
    #             "layer_image_attention_ratio": ratios.get("image", []),
    #             "layer_clinical_attention_ratio": ratios.get("clinical_total", []),
    #             # 细分指标
    #             "layer_dev_attention": ratios.get("dev", []),
    #             "layer_peri_attention": ratios.get("peri", []),
    #             "layer_phys_attention": ratios.get("phys", [])
    #         })
            
    #         # 强制垃圾回收，防止 CPU 内存溢出
    #         import gc
    #         gc.collect()

    #     # 全部跑完后，保存为 JSON
    #     if attention_data:
    #         with open(save_attn_path, "w", encoding="utf-8") as f:
    #             json.dump(attention_data, f, indent=4, ensure_ascii=False)
    #         print(f"\n所有样本的 Attention 数据已成功保存至: {save_attn_path}")

    #     return results
    