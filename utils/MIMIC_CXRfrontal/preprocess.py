"""
Build utils/MIMIC_CXRfrontal/test.json from MIMICCXR chat JSON.

Output rows are **full LLaVA-Rad-style chat objects** (same schema as the source):
conversations, image, id, view, impression, etc. No MedEvalKit legacy-only layout.

Filtering aligns with LLaVA-Rad `data_loader_mimic_cxr_all_frontal_findings`:
  - view in ('AP', 'PA')
  - conversations[1]['value'] is str (gpt findings)
Dedupe: one row per `image` path; tie-break by generate_method (same ranks as MIMIC_CXR).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any

from tqdm import tqdm

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_MEDEVALKIT_ROOT = os.path.abspath(os.path.join(_PKG_DIR, "..", ".."))


def _is_mimic_chat_row(sample: dict) -> bool:
    img = sample.get("image")
    if not isinstance(img, str):
        return False
    if "mimic/" not in img.replace("\\", "/"):
        return False
    return isinstance(sample.get("conversations"), list)


def _frontal_llavarad_ok(sample: dict) -> bool:
    """Same gate as llava/utils.py data_loader_mimic_cxr_all_frontal_findings."""
    conv = sample.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2:
        return False
    if not isinstance(conv[1].get("value"), str):
        return False
    view = sample.get("view")
    return view in ("AP", "PA")


def _image_key(sample: dict) -> str:
    return (sample.get("image") or "").replace("\\", "/").strip()


def _dedupe_full_rows(rows: list[dict]) -> list[dict]:
    """One row per image path; prefer generate_method like MIMIC_CXR _dedupe_chat_legacy."""
    prefer = (os.environ.get("MIMIC_CXR_GENERATE_METHOD") or "").strip().lower()

    def rank_meta(raw: dict) -> int:
        gm = (raw.get("generate_method") or "").strip().lower()
        if prefer and gm == prefer:
            return 0
        if gm == "rule-based":
            return 1
        if gm == "gpt4":
            return 2
        return 3

    by_key: dict[str, tuple[int, dict]] = {}
    order: list[str] = []

    for raw in rows:
        key = _image_key(raw)
        if not key:
            continue
        rk = rank_meta(raw)
        if key not in by_key:
            by_key[key] = (rk, raw)
            order.append(key)
        else:
            old_rk, _ = by_key[key]
            if rk < old_rk:
                by_key[key] = (rk, raw)
    return [copy.deepcopy(by_key[k][1]) for k in order]


def _default_chat_json() -> str:
    return os.path.join(
        _MEDEVALKIT_ROOT,
        "utils",
        "MIMIC-CXRfrontal",
        "MIMICCXR",
        "chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json",
    )


def _default_output_path() -> str:
    return os.path.join(_PKG_DIR, "test.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess MIMICCXR chat JSON → MIMIC_CXRfrontal/test.json (LLaVA-Rad chat rows, AP/PA)."
    )
    parser.add_argument(
        "--input",
        default=os.environ.get("MIMIC_CXRFRONTAL_PREPROCESS_INPUT", "").strip()
        or _default_chat_json(),
        help="Source chat JSON.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("MIMIC_CXRFRONTAL_PREPROCESS_OUTPUT", "").strip()
        or _default_output_path(),
        help="Destination test.json (list of full chat dicts).",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"[preprocess] input not found: {args.input}", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        dataset: list[Any] = json.load(f)

    if not isinstance(dataset, list):
        print("[preprocess] expected top-level JSON array", file=sys.stderr)
        return 1

    gen_f = (os.environ.get("MIMIC_CXR_GENERATE_METHOD") or "").strip().lower()
    candidates: list[dict] = []

    for sample in tqdm(dataset, desc="MIMIC_CXRfrontal preprocess"):
        if not isinstance(sample, dict):
            continue
        if not _is_mimic_chat_row(sample):
            continue
        if gen_f and (sample.get("generate_method") or "").strip().lower() != gen_f:
            continue
        if not _frontal_llavarad_ok(sample):
            continue
        gpt = ""
        for turn in sample.get("conversations") or []:
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                gpt = (turn.get("value") or "").strip()
        imp = (sample.get("impression") or "").strip()
        if gpt == "" and imp == "":
            continue
        candidates.append(sample)

    out_rows = _dedupe_full_rows(candidates)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    print(f"[preprocess] wrote {len(out_rows)} LLaVA-Rad-format rows → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
