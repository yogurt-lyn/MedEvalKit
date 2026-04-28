import torch
import os
import sys

import cv2
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
    

    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
             attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
        
        # [修改点 1] 强制返回 8 个值，补齐最后 3 个为 None
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

    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    def compute_layerwise_modality_attention(self, all_step_attentions, test_modal_token_spans):
        """
        计算指定 step 范围内，各层对 Image 和 Clinic Token 的平均关注率
        返回: (image_ratio_list, clinic_ratio_list)
        """
        import numpy as np

        if not all_step_attentions or len(all_step_attentions) == 0:
            return [], []

        generated_len = len(all_step_attentions)
        first_step = all_step_attentions[0]
        num_layers = len(first_step)
        batch_size = first_step[0].shape[0]

        layer_image_ratio = torch.zeros(num_layers)
        layer_clinical_ratio = torch.zeros(num_layers)

        for step_data in all_step_attentions:
            for layer_idx in range(num_layers):
                layer_attn = step_data[layer_idx]  # [B, Heads, 1, Seq]
                
                # 对 heads 平均
                attn = layer_attn.mean(dim=1)      # [B, 1, Seq]
                
                per_sample_image = []
                per_sample_clinical = []

                for b in range(batch_size):
                    single_attn = layer_attn[b].mean(dim=0)  # [1, Seq]
                    total = single_attn[0].sum().item() + 1e-8
                    spans = test_modal_token_spans[b]

                    # image
                    s, e = spans.get("image", (0, 0))
                    img_w = single_attn[0, s:e].sum().item() if e > s else 0.0

                    # clinical
                    cli_w = 0.0
                    for key in ["dev", "peri", "phys"]:
                        s, e = spans.get(key, (0, 0))
                        if e > s:
                            cli_w += single_attn[0, s:e].sum().item()

                    per_sample_image.append(img_w / total)
                    per_sample_clinical.append(cli_w / total)

                # batch 平均 (当前逻辑中通常 batch=1)
                layer_image_ratio[layer_idx] += np.mean(per_sample_image)
                layer_clinical_ratio[layer_idx] += np.mean(per_sample_clinical)

        # step 平均
        layer_image_ratio /= generated_len
        layer_clinical_ratio /= generated_len

        # 转为 list 以便保存为 JSON
        return layer_image_ratio.numpy().tolist(), layer_clinical_ratio.numpy().tolist()

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

    import cv2
    import numpy as np
    import os
    import json
    import math

    def generate_output(self, messages, output_attention_path=None):
        """生成输出并可选保存 I_mask 阶段的注意力热力图"""
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

        input_token_len = inputs['input_ids'].shape[1]

        with torch.inference_mode():
            try:  # <--- 最外层的 try
                outputs = self.model.generate(**generation_kwargs)
                
                modal_token_spans = GLOBAL_CACHE["spans"]
                
                if outputs.attentions is not None:
                    # 将所有注意力转移到 CPU 以节省显存
                    cpu_attentions = []
                    for step_attn in outputs.attentions:
                        layers_on_cpu = [layer.detach().cpu() for layer in step_attn]
                        cpu_attentions.append(layers_on_cpu)
                else:
                    cpu_attentions = None
                
                output_ids = outputs.sequences.detach().cpu()

                del outputs 
                torch.cuda.empty_cache() 

                attention_stats = {}

                if modal_token_spans is not None and cpu_attentions is not None:
                    # 1. 原始逻辑：计算全部 token 的统计
                    all_img_ratio, all_cli_ratio = self.compute_layerwise_modality_attention(
                        cpu_attentions, 
                        modal_token_spans
                    )
                    attention_stats["all_tokens"] = {
                        "image_ratio": all_img_ratio,
                        "clinic_ratio": all_cli_ratio
                    }

                    # 2. 原始逻辑：计算 D_mask 区域统计
                    DISEASE_DIAGNOSIS_IDS = [559, 24876, 19263, 29901] 
                    full_ids_list = output_ids[0].tolist()
                    
                    d_marker_pos = find_subseq_ids(full_ids_list, DISEASE_DIAGNOSIS_IDS)
                    d_content_start_idx = d_marker_pos[1] if d_marker_pos is not None else input_token_len
                        
                    d_step_start = max(0, d_content_start_idx - input_token_len)
                    d_step_end = len(cpu_attentions)
                    
                    d_mask_attentions = cpu_attentions[d_step_start:d_step_end] if d_step_start < d_step_end else cpu_attentions
                    d_img_ratio, d_cli_ratio = self.compute_layerwise_modality_attention(
                        d_mask_attentions, 
                        modal_token_spans
                    )
                    attention_stats["d_mask_tokens"] = {
                        "image_ratio": d_img_ratio,
                        "clinic_ratio": d_cli_ratio
                    }

                    # ==========================================================
                    # 3. [修正版] 严格在生成的 Token 中提取 I_mask 并生成热力图
                    # ==========================================================
                    if output_attention_path is not None and modal_token_spans[0].get("image") is not None:
                        try:  # <--- 内层的 try (用于捕获热力图生成错误)
                            IMAGING_CONCLUSION_IDS = [6751, 15997, 29901]
                            DISEASE_DIAGNOSIS_IDS  = [559, 24876, 19263, 29901]
                            
                            # 纯粹提取新生成的 tokens
                            generated_ids = output_ids[0, input_token_len:].tolist()
                            
                            # 在新生成的 tokens 中寻找 Marker
                            i_marker_pos_gen = find_subseq_ids(generated_ids, IMAGING_CONCLUSION_IDS)
                            d_marker_pos_gen = find_subseq_ids(generated_ids, DISEASE_DIAGNOSIS_IDS)
                            
                            # 直接映射为 attention 的 step 索引
                            if i_marker_pos_gen is not None:
                                i_step_start = i_marker_pos_gen[1] 
                            else:
                                i_step_start = 0  
                                
                            if d_marker_pos_gen is not None:
                                i_step_end = d_marker_pos_gen[0]
                            else:
                                i_step_end = len(generated_ids) 

                            # 安全越界截断
                            i_step_start = max(0, min(i_step_start, len(cpu_attentions)))
                            i_step_end = max(0, min(i_step_end, len(cpu_attentions)))

                            if i_step_start < i_step_end and len(cpu_attentions) > 0:
                                i_mask_attentions = cpu_attentions[i_step_start:i_step_end]
                                
                                if len(i_mask_attentions) == 0:
                                    print(f"[-] 警告: 实际提取到的 i_mask_attentions 为空，跳过: {output_attention_path}")
                                else:
                                    img_start, img_end = modal_token_spans[0]["image"]
                                    num_image_tokens = img_end - img_start
                                    
                                    total_img_attn = torch.zeros(num_image_tokens, dtype=torch.float32)

                                    for step_attn in i_mask_attentions:
                                        last_layer_attn = step_attn[-1].float() 
                                        attn_mean_heads = last_layer_attn.mean(dim=1)[0, -1, :] 
                                        img_attn = attn_mean_heads[img_start:img_end]
                                        total_img_attn += img_attn
                                    
                                    total_img_attn /= len(i_mask_attentions)
                                    
                                    grid_size = int(math.sqrt(num_image_tokens))
                                    expected_tokens = grid_size * grid_size
                                    
                                    if num_image_tokens != expected_tokens:
                                        attn_to_view = total_img_attn[:expected_tokens]
                                    else:
                                        attn_to_view = total_img_attn
                                        
                                    attn_2d = attn_to_view.view(grid_size, grid_size).numpy()
                                    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
                                    
                                    orig_image = messages.get("images", messages.get("image"))
                                    if isinstance(orig_image, list):
                                        orig_image = orig_image[0]
                                        
                                    if isinstance(orig_image, str):
                                        orig_image_cv = cv2.imread(orig_image)
                                        if orig_image_cv is None:
                                            raise ValueError(f"cv2 无法读取路径图像: {orig_image}")
                                    elif isinstance(orig_image, Image.Image):
                                        orig_image_cv = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
                                    else:
                                        raise ValueError("无法解析原始图像类型")
                                        
                                    heatmap = cv2.resize(attn_2d, (orig_image_cv.shape[1], orig_image_cv.shape[0]))
                                    heatmap = np.uint8(255 * heatmap)
                                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                                    
                                    superimposed_img = cv2.addWeighted(orig_image_cv, 0.6, heatmap, 0.4, 0)
                                    cv2.imwrite(output_attention_path, superimposed_img)
                                    print(f"[+] 成功保存 I_mask 热力图至: {output_attention_path}")
                        
                        except Exception as e: # <--- 内层 try 的 except
                            import traceback
                            print(f"[!] I_mask 注意力可视化失败: {e}")
                            traceback.print_exc()

            except Exception as e: # <--- 最外层 try 的 except，之前你漏掉了这个
                print(f"生成失败: {e}")
                torch.cuda.empty_cache() 
                raise

        # <--- 注意这里，严格和 with torch.inference_mode(): 同级对齐
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], 
            skip_special_tokens=True
        )[0].strip()
        
        return generated_text, attention_stats
        
    
    def generate_outputs(self, messages_list, save_json_path="attention_recordsllavaradImask.json", output_dir="/data_lyn/heatmapsImaskllavarad"):
        """
        批量生成输出，保存注意力 JSON 记录，并批量保存 I_mask 注意力热力图
        """
        results = []
        all_attention_records = []
        
        # 自动创建热力图输出目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for idx, messages in enumerate(messages_list):
            # 动态生成 heatmap_0.png, heatmap_1.png...
            attn_path = os.path.join(output_dir, f"heatmap_{idx}.png") if output_dir else None
            
            # 将路径传给 generate_output
            text, stats = self.generate_output(messages, output_attention_path=attn_path)
            results.append(text)
            
            if stats:
                stats["sample_index"] = idx
                stats["generated_text"] = text
                all_attention_records.append(stats)
                
        if all_attention_records and save_json_path:
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(all_attention_records, f, indent=4, ensure_ascii=False)
            print(f">>> [Saved] 所有注意力关注率已成功保存至: {save_json_path}")
            
        return results

    # def generate_output(self, messages):
    #     """生成输出"""
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
    #         "output_attentions": True, # 必须开启
    #         "return_dict_in_generate": True,
    #     }

    #     # 提前计算 input 长度，供后续步数切片使用
    #     input_token_len = inputs['input_ids'].shape[1]

    #     with torch.inference_mode():
    #         try:
    #             outputs = self.model.generate(**generation_kwargs)
                
    #             modal_token_spans = GLOBAL_CACHE["spans"]
                
    #             if modal_token_spans is not None:
    #                 print(">>> [SUCCESS] 成功从全局缓存拿回 last_modal_spans!")
    #             else:
    #                 print(">>> [FAIL] 全局缓存为空！可能是 prepare 没有正确执行 prefill 逻辑。")

    #             if outputs.attentions is not None:
    #                 cpu_attentions = []
    #                 for step_attn in outputs.attentions:
    #                     layers_on_cpu = [layer.detach().cpu() for layer in step_attn]
    #                     cpu_attentions.append(layers_on_cpu)
    #             else:
    #                 cpu_attentions = None
                
    #             output_ids = outputs.sequences.detach().cpu()

    #             del outputs 
    #             torch.cuda.empty_cache() 

    #             # 用来存储这单条数据的注意力统计
    #             attention_stats = {}

    #             if modal_token_spans is not None and cpu_attentions is not None:
    #                 # ==========================================================
    #                 # 1. 计算【所有 Output Token】的关注率
    #                 # ==========================================================
    #                 all_img_ratio, all_cli_ratio = self.compute_layerwise_modality_attention(
    #                     cpu_attentions, 
    #                     modal_token_spans
    #                 )
    #                 attention_stats["all_tokens"] = {
    #                     "image_ratio": all_img_ratio,
    #                     "clinic_ratio": all_cli_ratio
    #                 }

    #                 # ==========================================================
    #                 # 2. 定位 D_mask 区域，计算【D_mask 区域】的关注率
    #                 # ==========================================================
    #                 DISEASE_DIAGNOSIS_IDS = [559, 24876, 19263, 29901] 
    #                 full_ids_list = output_ids[0].tolist()
                    
    #                 d_marker_pos = find_subseq_ids(full_ids_list, DISEASE_DIAGNOSIS_IDS)
                    
    #                 if d_marker_pos is not None:
    #                     d_content_start_idx = d_marker_pos[1]
    #                 else:
    #                     d_content_start_idx = input_token_len
                        
    #                 step_start = max(0, d_content_start_idx - input_token_len)
    #                 step_end = len(cpu_attentions)
                    
    #                 if step_start < step_end:
    #                     d_mask_attentions = cpu_attentions[step_start:step_end]
    #                 else:
    #                     d_mask_attentions = cpu_attentions
                    
    #                 # 传入切片过滤后的 attentions 计算 D_mask 关注率
    #                 d_img_ratio, d_cli_ratio = self.compute_layerwise_modality_attention(
    #                     d_mask_attentions, 
    #                     modal_token_spans
    #                 )
    #                 attention_stats["d_mask_tokens"] = {
    #                     "image_ratio": d_img_ratio,
    #                     "clinic_ratio": d_cli_ratio
    #                 }
                
    #         except Exception as e:
    #             print(f"生成失败: {e}")
    #             torch.cuda.empty_cache() 
    #             raise
        
    #     generated_text = self.tokenizer.batch_decode(
    #         output_ids[:, input_token_len:], 
    #         skip_special_tokens=True
    #     )[0].strip()
        
    #     # 返回元组: (生成的文本, 注意力统计字典)
    #     return generated_text, attention_stats
    
    # def generate_outputs(self, messages_list, save_json_path="attention_records.json"):
    #     """
    #     批量生成输出，并将注意力统计数据保存到 JSON 文件
    #     """
    #     results = []
    #     all_attention_records = []
        
    #     import json
        
    #     for idx, messages in enumerate(messages_list):
    #         # 接收文本和统计字典
    #         text, stats = self.generate_output(messages)
    #         results.append(text)
            
    #         # 如果成功获取了注意力统计，加入记录列表
    #         if stats:
    #             stats["sample_index"] = idx
    #             stats["generated_text"] = text
    #             all_attention_records.append(stats)
                
    #     # 批量处理结束后，保存 JSON 文件
    #     if all_attention_records and save_json_path:
    #         with open(save_json_path, "w", encoding="utf-8") as f:
    #             json.dump(all_attention_records, f, indent=4, ensure_ascii=False)
    #         print(f">>> [Saved] 所有注意力关注率已成功保存至: {save_json_path}")
            
    #     return results
    