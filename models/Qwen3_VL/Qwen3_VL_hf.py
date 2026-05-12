import os
from pathlib import Path
import importlib

import torch
import torch.nn.functional as F


def _qwen3_vl_for_conditional_generation_cls():
    """Resolve Qwen3VLForConditionalGeneration (older transformers only export via submodule)."""
    try:
        from transformers import Qwen3VLForConditionalGeneration as _Cls

        return _Cls
    except ImportError:
        pass
    try:
        m = importlib.import_module("transformers.models.qwen3_vl.modeling_qwen3_vl")
        return m.Qwen3VLForConditionalGeneration
    except (ImportError, AttributeError) as exc:
        import transformers as _tf

        ver = getattr(_tf, "__version__", "?")
        raise ImportError(
            "Qwen3-VL 需要包含 Qwen3VLForConditionalGeneration 的 Hugging Face transformers "
            f"(一般 >=4.57；当前 transformers=={ver})。请执行: "
            'pip install -U "transformers>=4.57.0" huggingface_hub'
        ) from exc


Qwen3VLForConditionalGeneration = _qwen3_vl_for_conditional_generation_cls()
from transformers import AutoConfig, AutoProcessor
from qwen_vl_utils import process_vision_info

from models.pca_cknna_viz import save_sample_pca_hf
from models.decoder_self_attn_viz import (
    find_vision_token_range,
    save_decoder_self_attn,
)


def _clean_image_placeholders(prompt):
    lines = []
    for line in str(prompt or "").splitlines():
        stripped = line.strip()
        if stripped == "<image>" or (
            stripped.startswith("<image_") and stripped.endswith(">:")
        ):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _bool_arg(args, name, default="False"):
    return str(getattr(args, name, default) or default).lower() in ("1", "true", "yes")


def _unwrap_qwen_image_features_output(raw):
    """
    Qwen3VLModel.get_image_features 在不同 transformers 版本下可能返回
    ModelOutput 或 (ModelOutput,) 等 tuple；统一解包成带 pooler_output 的对象。
    """
    cur = raw
    for _ in range(4):
        if hasattr(cur, "pooler_output"):
            return cur
        if isinstance(cur, (tuple, list)) and len(cur) > 0:
            cur = cur[0]
            continue
        break
    raise TypeError(
        f"Unexpected Qwen3-VL get_image_features return type after unwrap: {type(raw)} -> {type(cur)}"
    )


def _patch_qwen3_vl_text_rope_scaling(config):
    """
    verl / merge 脚本导出的 config 里 text_config.rope_scaling 常为 null；
    transformers>=4.57 的 Qwen3VLTextRotaryEmbedding 会对其调用 .get() 导致崩溃。
    缺省时补成与官方 Qwen3-VL-8B 一致的 MRoPE 字段。
    """
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        return config
    rs = getattr(text_cfg, "rope_scaling", None)
    if isinstance(rs, dict):
        return config
    default_rs = {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    }
    setattr(text_cfg, "rope_scaling", default_rs)
    return config


class Qwen3_VL:
    def __init__(self, model_path, args):
        super().__init__()

        self._self_attn_viz_enabled = _bool_arg(args, "dq_self_attn_visualize", "False")
        # eager 才能拿 attentions；flash-attn / sdpa 都不返回 attentions。
        attn_impl = "eager" if self._self_attn_viz_enabled else "flash_attention_2"

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        cfg = _patch_qwen3_vl_text_rope_scaling(cfg)

        self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        # verl / merge 导出的 tokenizer_config 常把 extra_special_tokens 写成 list；
        # transformers 新版在 PreTrainedTokenizerBase 里对其调用 .keys()，会触发
        # AttributeError: 'list' object has no attribute 'keys'。
        # from_pretrained 合并参数时用户 kwargs 覆盖 config，故用空 dict 覆盖错误字段即可；
        # 多模态 special tokens 仍来自 vocab / added_tokens_decoder。
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            extra_special_tokens={},
        )

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

        # ---- decoder self-attn 可视化状态（与 LLava_Rad 同款）----
        raw_dir = (getattr(args, "dq_self_attn_visualize_dir", None) or "").strip()
        if raw_dir:
            self._self_attn_viz_root = os.path.expanduser(raw_dir)
        else:
            self._self_attn_viz_root = os.path.join(
                os.path.expanduser(getattr(args, "output_path", ".")),
                "self_attn_viz",
            )
        layer_str = str(getattr(args, "dq_self_attn_visualize_layers", "") or "").strip()
        self._self_attn_viz_layers = (
            {int(x) for x in layer_str.split(",") if x.strip() != ""} if layer_str else None
        )
        _agg = str(getattr(args, "dq_self_attn_visualize_agg", "last") or "last").strip().lower()
        self._self_attn_agg = "mean" if _agg in ("mean", "mean_nonvis", "mean_text") else "last"
        self._self_attn_viz_forward_id = 0
        if self._self_attn_viz_enabled:
            os.makedirs(self._self_attn_viz_root, exist_ok=True)
            print(
                "[Qwen3-VL] dq_self_attn_visualize=ON "
                f"dir={self._self_attn_viz_root} "
                f"layers={sorted(self._self_attn_viz_layers) if self._self_attn_viz_layers else 'all'} "
                f"agg={self._self_attn_agg} attn_impl={attn_impl}"
            )

    def process_messages(self, messages):
        new_messages = []
        if "system" in messages:
            new_messages.append({"role": "system", "content": messages["system"]})
        if "image" in messages:
            prompt = _clean_image_placeholders(messages["prompt"])
            new_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image", "image": messages["image"]}, {"type": "text", "text": prompt}],
                }
            )
        elif "images" in messages:
            prompt = _clean_image_placeholders(messages["prompt"])
            content = []
            for image in messages["images"]:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            new_messages.append({"role": "user", "content": content})
        else:
            new_messages.append({"role": "user", "content": [{"type": "text", "text": messages["prompt"]}]})
        messages = new_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return inputs

    @torch.inference_mode()
    def _save_decoder_self_attn_prefill(self, inputs):
        """与 LLava_Rad._save_decoder_self_attn_last_query 同款：跑一次 prefill，画 decoder 各层每头的 vision attention。"""
        forward_inputs = {
            k: v
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        }
        outputs = self.llm(
            **forward_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        att_tuple = getattr(outputs, "attentions", None)
        if not att_tuple:
            print("[Qwen3-VL self-attn viz] skip: attentions unavailable")
            return

        image_token_id = int(getattr(self.llm.config, "image_token_id"))
        rng = find_vision_token_range(inputs["input_ids"][0], image_token_id)
        if rng is None:
            print("[Qwen3-VL self-attn viz] skip: no image_token_id in input_ids")
            return
        vis_lo, vis_hi = rng
        n_vis = vis_hi - vis_lo

        side_h = side_w = None
        grid = inputs.get("image_grid_thw")
        try:
            sm = int(self.llm.model.visual.spatial_merge_size)
        except AttributeError:
            sm = 2
        if grid is not None and len(grid) > 0:
            h = int(grid[0, 1].item()) // sm
            w = int(grid[0, 2].item()) // sm
            if h * w == n_vis and h > 0 and w > 0:
                side_h, side_w = h, w

        self._self_attn_viz_forward_id += 1
        out_root = Path(self._self_attn_viz_root) / f"forward_{self._self_attn_viz_forward_id:05d}"
        saved = save_decoder_self_attn(
            att_tuple,
            vis_lo,
            vis_hi,
            out_root,
            agg=self._self_attn_agg,
            layers_filter=self._self_attn_viz_layers,
            side_h=side_h,
            side_w=side_w,
        )
        print(
            f"[Qwen3-VL self-attn viz] saved {saved} layers under {out_root} "
            f"(grid={side_h}x{side_w}, n_vis={n_vis}, agg={self._self_attn_agg})"
        )

    def generate_output(self, messages):
        inputs = self.process_messages(messages)
        if self._self_attn_viz_enabled:
            try:
                self._save_decoder_self_attn_prefill(inputs)
            except Exception as exc:
                print(f"[Qwen3-VL self-attn viz] failed: {exc}")
        do_sample = False if self.temperature == 0 else True
        generated_ids = self.llm.generate(
            **inputs,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def generate_outputs(self, messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res

    @torch.inference_mode()
    def _text_embedding(self, text):
        toks = self.processor.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        emb = self.llm.get_input_embeddings()
        token_ids = toks["input_ids"].to(self.llm.device)
        e = emb(token_ids).squeeze(0)
        return F.normalize(e.mean(dim=0), dim=0)

    @torch.inference_mode()
    def _text_token_embeddings(self, text):
        toks = self.processor.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        emb = self.llm.get_input_embeddings()
        token_ids = toks["input_ids"].to(self.llm.device)
        e = emb(token_ids).squeeze(0)
        e = torch.nan_to_num(e.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        return F.normalize(e, dim=-1)

    @torch.inference_mode()
    def _extract_global_projected_features(self, inputs):
        raw = self.llm.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        out = _unwrap_qwen_image_features_output(raw)
        merged = out.pooler_output
        if isinstance(merged, (list, tuple)):
            feat = merged[0]
        else:
            feat = merged
        return [feat.unsqueeze(0)]

    @torch.inference_mode()
    def _extract_local_projected_features(self, inputs):
        raw = self.llm.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        out = _unwrap_qwen_image_features_output(raw)
        merge_size = int(self.llm.model.visual.spatial_merge_size)
        grid = inputs["image_grid_thw"]
        split_sizes = (grid.prod(-1) // (merge_size**2)).tolist()

        feats_list = []
        deep = getattr(out, "deepstack_features", None) or []
        for df in deep:
            chunks = torch.split(df, split_sizes)
            first = chunks[0]
            feats_list.append(first.unsqueeze(0))

        merged = out.pooler_output
        if isinstance(merged, (list, tuple)):
            final = merged[0]
        else:
            final = merged
        feats_list.append(final.unsqueeze(0))
        return feats_list

    def generate_outputs_with_pca(
        self, messages_list, pca_output_dir=None, pca_max_samples=8, sample_start_idx=0
    ):
        """供 BaseDataset 在 ENABLE_PCA_VIZ / ENABLE_CKNNA_VIZ 时调用（与 LLava_rad_hf 同款产物）。"""
        if pca_output_dir is None:
            pca_output_dir = os.path.join(".", "pca_viz")
        results = []
        for idx, messages in enumerate(messages_list):
            global_idx = sample_start_idx + idx
            response = self.generate_output(messages)
            results.append(response)
            if global_idx < pca_max_samples:
                try:
                    save_sample_pca_hf(self, messages, response, global_idx, pca_output_dir)
                except Exception as exc:
                    print(f"[PCA_VIZ] sample={global_idx} failed: {exc}")
        return results
