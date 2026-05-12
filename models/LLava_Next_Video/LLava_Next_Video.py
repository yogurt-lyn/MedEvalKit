import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

from models.pca_cknna_viz import save_sample_pca_hf
from models.decoder_self_attn_viz import (
    find_vision_token_range,
    save_decoder_self_attn,
)


def _import_image_size_to_num_patches():
    try:
        from transformers.models.llava_next_video.modeling_llava_next_video import (
            image_size_to_num_patches,
        )

        return image_size_to_num_patches
    except Exception:
        return None


def _bool_arg(args, name, default="False"):
    return str(getattr(args, name, default) or default).lower() in ("1", "true", "yes")


class LlavaNextVideo:
    def __init__(self, model_path, args):
        super().__init__()

        self._self_attn_viz_enabled = _bool_arg(args, "dq_self_attn_visualize", "False")
        attn_impl = "eager" if self._self_attn_viz_enabled else None  # eager 才能拿 attentions

        self.processor = LlavaNextVideoProcessor.from_pretrained(model_path)

        kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl
        self.llm = LlavaNextVideoForConditionalGeneration.from_pretrained(model_path, **kwargs)
        self.llm.to("cuda")

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
                "[LLava_Next_Video] dq_self_attn_visualize=ON "
                f"dir={self._self_attn_viz_root} "
                f"layers={sorted(self._self_attn_viz_layers) if self._self_attn_viz_layers else 'all'} "
                f"agg={self._self_attn_agg} attn_impl={attn_impl}"
            )

    def process_messages(self, messages):
        """处理输入消息，转换为LLaVA-NeXT格式"""
        prompt = ""
        images = None

        if "system" in messages:
            prompt = messages["system"]

        prompt = prompt + "\n" + messages["prompt"]

        if "images" in messages:
            images = messages["images"]
            if not isinstance(images, list):
                images = [images]
        elif "image" in messages:
            image = messages["image"]
            if isinstance(image, str):
                image = Image.open(image)
            images = [image]

        content = [{"type": "text", "text": prompt}]

        if images:
            for _img in images:
                content.append({"type": "image"})

        conversation = [{"role": "user", "content": content}]

        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        if images:
            inputs = self.processor(
                text=formatted_prompt,
                images=images,
                return_tensors="pt",
            ).to(self.llm.device)
        else:
            inputs = self.processor(
                text=formatted_prompt,
                return_tensors="pt",
            ).to(self.llm.device)

        return inputs

    @torch.inference_mode()
    def _save_decoder_self_attn_prefill(self, inputs):
        """与 LLava_Rad._save_decoder_self_attn_last_query 同款：跑一次 prefill，画 decoder 各层每头的 vision attention。"""
        forward_inputs = {
            k: v
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "pixel_values", "image_sizes")
        }
        outputs = self.llm(
            **forward_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        att_tuple = getattr(outputs, "attentions", None)
        if not att_tuple:
            print("[LLava_Next_Video self-attn viz] skip: attentions unavailable")
            return

        image_token_id = int(getattr(self.llm.config, "image_token_id"))
        rng = find_vision_token_range(inputs["input_ids"][0], image_token_id)
        if rng is None:
            print("[LLava_Next_Video self-attn viz] skip: no image_token_id in input_ids")
            return
        vis_lo, vis_hi = rng
        n_vis = vis_hi - vis_lo

        self._self_attn_viz_forward_id += 1
        out_root = Path(self._self_attn_viz_root) / f"forward_{self._self_attn_viz_forward_id:05d}"
        saved = save_decoder_self_attn(
            att_tuple,
            vis_lo,
            vis_hi,
            out_root,
            agg=self._self_attn_agg,
            layers_filter=self._self_attn_viz_layers,
        )
        print(
            f"[LLava_Next_Video self-attn viz] saved {saved} layers under {out_root} "
            f"(n_vis={n_vis}, agg={self._self_attn_agg})"
        )

    def generate_output(self, messages):
        """生成输出"""
        inputs = self.process_messages(messages)

        if self._self_attn_viz_enabled:
            try:
                self._save_decoder_self_attn_prefill(inputs)
            except Exception as exc:
                print(f"[LLava_Next_Video self-attn viz] failed: {exc}")

        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True if self.temperature > 0 else False,
        }

        with torch.no_grad():
            outputs = self.llm.generate(**inputs, **generation_config)

        generated_text = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return generated_text.strip()

    def generate_outputs(self, messages_list):
        """批量生成输出"""
        results = []
        for messages in messages_list:
            result = self.generate_output(messages)
            results.append(result)
        return results

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

    def _prepare_lnv_pixel_values(self, pixel_values, image_sizes):
        """与 LlavaNextVideoModel.get_image_features 一致：将 5D batch 展平为 4D patch 序列。"""
        image_size_to_num_patches = _import_image_size_to_num_patches()
        if image_size_to_num_patches is None:
            raise RuntimeError("无法导入 image_size_to_num_patches，请升级 transformers。")

        grid_pinpoints = self.llm.config.image_grid_pinpoints
        patch_size = self.llm.config.vision_config.image_size
        image_num_patches = [
            image_size_to_num_patches(imsize, grid_pinpoints, patch_size) for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            _pixel_values_list = [
                pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values shape {pixel_values.shape}, expect 4d or 5d")

        return pixel_values, image_num_patches

    @torch.inference_mode()
    def _extract_global_projected_features(self, inputs):
        out = self.llm.model.get_image_features(
            inputs["pixel_values"],
            inputs["image_sizes"],
        )
        pool = out.pooler_output
        if isinstance(pool, (list, tuple)):
            feat = pool[0]
        else:
            feat = pool
        return [feat.unsqueeze(0)]

    @torch.inference_mode()
    def _extract_local_projected_features(self, inputs):
        pixel_values, image_num_patches = self._prepare_lnv_pixel_values(
            inputs["pixel_values"], inputs["image_sizes"]
        )
        strategy = getattr(self.llm.config, "vision_feature_select_strategy", None) or "default"

        vo = self.llm.model.vision_tower(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hs_list = vo.hidden_states
        feats_list = []
        for li in range(len(hs_list)):
            selected = hs_list[li][:, 1:]
            projected = self.llm.model.multi_modal_projector(selected)
            projected_split = torch.split(projected, image_num_patches, dim=0)
            image_features = list(projected_split)
            packed, _ = self.llm.model.pack_image_features(
                image_features,
                inputs["image_sizes"],
                strategy,
                image_newline=self.llm.model.image_newline,
            )
            feats_list.append(packed[0].unsqueeze(0))
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
