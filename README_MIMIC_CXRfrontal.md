# MIMIC_CXRfrontal Evaluation

This repo contains code for evaluating models on the frontal AP/PA subset of MIMIC-CXR in MedEvalKit. It does not commit MIMIC-CXR data or generated `test.json`; create those files on each machine from data you are authorized to use.

## Prepare The Dataset

From the repo root:

```bash
export MEDEVALKIT_ROOT="$PWD"
export MIMIC_CXRFRONTAL_PREPROCESS_INPUT=/path/to/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json
python -m utils.MIMIC_CXRfrontal.preprocess
```

This writes `utils/MIMIC_CXRfrontal/test.json`. The evaluator expects image paths under the MIMIC-CXR-JPG `files/` directory. If your image root is not the default MedEvalKit layout, set:

```bash
export DQLLAVA_MIMIC_JPG_ROOT=/path/to/mimic-cxr-jpg/2.0.0/files
```

Optional filters:

```bash
export MIMIC_CXR_GENERATE_METHOD=rule-based
export MIMIC_CXRFRONTAL_PREPROCESS_OUTPUT=/path/to/test.json
```

## Run Locally

```bash
export EVAL_DATASETS=MIMIC_CXRfrontal
export DATASETS_PATH="$PWD/utils"
export OUTPUT_PATH="$PWD/eval_results/mimic_cxr_frontal_qwen3vl"
export MODEL_NAME=Qwen3-VL
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export USE_VLLM=False
export ENABLE_PCA_VIZ=False
export ENABLE_CKNNA_VIZ=False
bash eval.sh
```

For Qwen2.5-VL use `MODEL_NAME=Qwen2.5-VL`. For LLava-Rad use `MODEL_NAME=LLava_Rad` and set `MODEL_PATH`, `LLAVA_RAD_WEIGHT_ROOT`, and `LLAVA_RAD_BASE_MODEL_PATH` for your host.

## Slurm Examples

The repository includes portable examples:

```bash
sbatch sbatch_eval_qwen3vl_mimic_cxr_frontal_l40.sh
sbatch sbatch_eval_qwen25vl_mimic_cxr_frontal_l40.sh
sbatch sbatch_eval_llavarad_mimic_cxr_frontal_l40.sh
sbatch sbatch_eval_llavarad_mimic_cxr_frontal_l40_joint_ckpt.sh
```

Override paths at submission time:

```bash
sbatch --export=ALL,MEDEVALKIT_ROOT="$PWD",MODEL_PATH=/path/to/model,DQLLAVA_MIMIC_JPG_ROOT=/path/to/mimic-cxr-jpg/2.0.0/files sbatch_eval_qwen3vl_mimic_cxr_frontal_l40.sh
```

Metrics are written under `OUTPUT_PATH/MODEL_NAME/MIMIC_CXRfrontal/`. Report-generation metrics can then be computed with the existing scripts under `utils/Metrics_Compute/`.
