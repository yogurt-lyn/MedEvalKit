import os
import json

from PIL import Image
from tqdm import tqdm

from ..base_dataset import BaseDataset
from ..question_formats import get_report_generation_prompt


def _mimic_jpg_root() -> str:
    v = (os.environ.get("DQLLAVA_MIMIC_JPG_ROOT") or "").strip()
    if v:
        return v
    mede = (os.environ.get("MEDEVALKIT_ROOT") or "").strip()
    if mede:
        return os.path.join(
            mede,
            "utils",
            "MIMIC_CXR",
            "MIMIC_CXR",
            "data",
            "wcl",
            "physionet.org",
            "files",
            "mimic-cxr-jpg",
            "2.0.0",
            "files",
        )
    _here = os.path.dirname(os.path.abspath(__file__))
    medeval_root = os.path.abspath(os.path.join(_here, "..", ".."))
    return os.path.join(
        medeval_root,
        "utils",
        "MIMIC_CXR",
        "MIMIC_CXR",
        "data",
        "wcl",
        "physionet.org",
        "files",
        "mimic-cxr-jpg",
        "2.0.0",
        "files",
    )


def _is_llavarad_chat_row(sample: dict) -> bool:
    return (
        isinstance(sample.get("conversations"), list)
        and isinstance(sample.get("image"), str)
        and "mimic/" in sample["image"].replace("\\", "/")
    )


def _human_prompt(sample: dict) -> str:
    for turn in sample.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") == "human":
            return (turn.get("value") or "").strip()
    return ""


def _gpt_findings(sample: dict) -> str:
    for turn in sample.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") == "gpt":
            return (turn.get("value") or "").strip()
    return ""


def _open_mimic_image(jpg_root: str, image_field: str) -> Image.Image:
    """Map mimic/pXX/... path under DQLLAVA jpg root (same layout as MedEvalKit MIMIC jpg tree)."""
    rel = (image_field or "").replace("\\", "/").strip()
    if rel.startswith("mimic/"):
        rel = rel[len("mimic/") :]
    path = os.path.join(jpg_root, rel)
    return Image.open(path).convert("RGB")


class MIMIC_CXRfrontal(BaseDataset):
    """
    Frontal MIMIC-CXR in **LLaVA-Rad chat row** format (test.json list of full dicts).

    Model input: first human `conversations[.*].value` (typically includes <image> line).
    Reference for metrics: gpt findings + top-level impression metadata (attached as
    `findings` / `impression` keys for cal_metrics / RRG scripts).
    """

    def __init__(self, model, dataset_path, output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))

    def load_data(self):
        json_path = os.path.join(self.dataset_path, "test.json")
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError(f"MIMIC_CXRfrontal: expected list in {json_path}")

        for idx, sample in enumerate(tqdm(dataset)):
            if not isinstance(sample, dict):
                continue
            if idx % self.num_chunks != self.chunk_idx:
                continue

            if _is_llavarad_chat_row(sample):
                findings = _gpt_findings(sample)
                impression = (sample.get("impression") or "").strip()
            else:
                # Legacy MedEvalKit rows (id, subject_id, AP_image, …) — still supported if present.
                findings = (sample.get("findings") or "").strip()
                impression = (sample.get("impression") or "").strip()

            if findings == "" and impression == "":
                continue

            if _is_llavarad_chat_row(sample):
                sample = dict(sample)
                sample["findings"] = findings
                sample["impression"] = impression

            sample = self.construct_messages(sample)
            self.samples.append(sample)

        max_s = int(os.environ.get("MAX_EVAL_SAMPLES", "0") or "0")
        if max_s > 0:
            self.samples = self.samples[:max_s]
            print("MIMIC_CXRfrontal: MAX_EVAL_SAMPLES cap applied:", max_s)
        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages(self, sample):
        jpg_root = _mimic_jpg_root()

        if _is_llavarad_chat_row(sample):
            prompt = _human_prompt(sample)
            if not prompt:
                raise ValueError("MIMIC_CXRfrontal: missing human conversation value")
            images = [_open_mimic_image(jpg_root, sample["image"])]
        else:
            subject_id = str(sample["subject_id"])
            study_id = str(sample["id"])
            name = sample["AP_image"]
            images = [
                Image.open(
                    os.path.join(
                        jpg_root,
                        f"p{subject_id[:2]}",
                        f"p{subject_id}",
                        f"s{study_id}",
                        f"{name}.jpg",
                    )
                ).convert("RGB")
            ]
            prompt = get_report_generation_prompt()

        messages = {"prompt": prompt, "images": images}
        sample["messages"] = messages
        return sample

    def cal_metrics(self, out_samples):
        import pandas as pd

        predictions_data = []
        ground_truth_data = []

        for i, sample in enumerate(out_samples):
            response = sample["response"]
            findings = sample["findings"]
            impression = sample["impression"]
            golden = f"Findings: {findings} Impression: {impression}."
            study_id = f"study_{i+1}"

            predictions_data.append({"study_id": study_id, "report": response})
            ground_truth_data.append({"study_id": study_id, "report": golden})

        predictions_df = pd.DataFrame(predictions_data)
        ground_truth_df = pd.DataFrame(ground_truth_data)

        prediction_path = os.path.join(self.output_path, "predictions.csv")
        ground_truth_path = os.path.join(self.output_path, "ground_truth.csv")
        predictions_df.to_csv(prediction_path, index=False)
        ground_truth_df.to_csv(ground_truth_path, index=False)

        return {"total metrics": "please use cal_report_metrics.py to generate metrics"}, out_samples
