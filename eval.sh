#!/usr/bin/env bash
set -euo pipefail

# Generic MedEvalKit evaluation launcher.
# Override any value below with environment variables, e.g.:
#   EVAL_DATASETS=MIMIC_CXRfrontal DATASETS_PATH="$PWD/utils" MODEL_NAME=Qwen3-VL \
#   MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct USE_VLLM=False bash eval.sh

PROJECT_ROOT="${MEDEVALKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_NAME="${CONDA_NAME:-}"

if [[ -n "${CONDA_NAME}" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_NAME}"
fi

cd "${PROJECT_ROOT}"
unset PYTORCH_CUDA_ALLOC_CONF

OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/eval_results/default}"
EVAL_DATASETS="${EVAL_DATASETS:-IU_XRAY}"
DATASETS_PATH="${DATASETS_PATH:-${PROJECT_ROOT}/utils/IU_XRAY}"
MODEL_NAME="${MODEL_NAME:-LLava_Rad}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/LLaVARadWeight}"

if [[ "${MODEL_NAME}" == "LLava_Rad" || "${MODEL_NAME}" == "DQLLava" || "${MODEL_NAME}" == "NeoRed" ]]; then
  LLAVA_RAD_WEIGHT_ROOT="${LLAVA_RAD_WEIGHT_ROOT:-${PROJECT_ROOT}/LLaVARadWeight}"
  export LLAVA_RAD_BASE_MODEL_PATH="${LLAVA_RAD_BASE_MODEL_PATH:-${LLAVA_RAD_WEIGHT_ROOT}/vicuna-7b-v1.5}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
USE_VLLM="${USE_VLLM:-False}"

SEED="${SEED:-42}"
REASONING="${REASONING:-False}"
TEST_TIMES="${TEST_TIMES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_IMAGE_NUM="${MAX_IMAGE_NUM:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.0001}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"

ENABLE_PCA_VIZ="${ENABLE_PCA_VIZ:-False}"
ENABLE_CKNNA_VIZ="${ENABLE_CKNNA_VIZ:-False}"
PCA_MAX_SAMPLES="${PCA_MAX_SAMPLES:-8}"
PCA_OUTPUT_SUBDIR="${PCA_OUTPUT_SUBDIR:-pca_viz}"
PCA_LAYER_IDS="${PCA_LAYER_IDS:-}"
PCA_VISUAL_SOURCE="${PCA_VISUAL_SOURCE:-global}"
CKNNA_TOPK="${CKNNA_TOPK:-10}"

USE_LLM_JUDGE="${USE_LLM_JUDGE:-False}"
GPT_MODEL="${GPT_MODEL:-None}"
JUDGE_MODEL_TYPE="${JUDGE_MODEL_TYPE:-openai}"
API_KEY="${API_KEY:-None}"
BASE_URL="${BASE_URL:-None}"

echo "[eval.sh] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[eval.sh] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-} PYTHON=$(command -v python || true)"
echo "[eval.sh] EVAL_DATASETS=${EVAL_DATASETS}"
echo "[eval.sh] DATASETS_PATH=${DATASETS_PATH}"
echo "[eval.sh] MODEL_NAME=${MODEL_NAME}"
echo "[eval.sh] MODEL_PATH=${MODEL_PATH}"
echo "[eval.sh] OUTPUT_PATH=${OUTPUT_PATH}"

python eval.py \
  --eval_datasets "${EVAL_DATASETS}" \
  --datasets_path "${DATASETS_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --seed "${SEED}" \
  --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --use_vllm "${USE_VLLM}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_image_num "${MAX_IMAGE_NUM}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --repetition_penalty "${REPETITION_PENALTY}" \
  --reasoning "${REASONING}" \
  --use_llm_judge "${USE_LLM_JUDGE}" \
  --judge_model_type "${JUDGE_MODEL_TYPE}" \
  --judge_model "${GPT_MODEL}" \
  --api_key "${API_KEY}" \
  --base_url "${BASE_URL}" \
  --test_times "${TEST_TIMES}" \
  --enable_pca_viz "${ENABLE_PCA_VIZ}" \
  --enable_cknna_viz "${ENABLE_CKNNA_VIZ}" \
  --pca_max_samples "${PCA_MAX_SAMPLES}" \
  --pca_output_subdir "${PCA_OUTPUT_SUBDIR}" \
  --pca_layer_ids "${PCA_LAYER_IDS}" \
  --pca_visual_source "${PCA_VISUAL_SOURCE}" \
  --cknna_topk "${CKNNA_TOPK}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}"
