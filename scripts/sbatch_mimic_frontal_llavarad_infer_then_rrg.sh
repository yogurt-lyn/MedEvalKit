#!/bin/bash
# 同一作业内：① MedEvalKit eval.py — LLava_Rad 跑 MIMIC_CXR（chat 测试集 + 仅 frontal AP/PA）
#           ② LLaVA-Rad RRG（run_rrg_from_results_json.sh）对刚生成的 results.json 打分。
#
# 提交示例：
#   sbatch /share/home/u16050/apps/Lynne/MLLM/MedEvalKit/scripts/sbatch_mimic_frontal_llavarad_infer_then_rrg.sh
#
# 常用环境变量覆盖：
#   MODEL_PATH  MODEL_NAME  RUN_TAG  OUTPUT_PATH  RESULT_ROOT
#   MIMIC_CXR_ANNOTATION_JSON  MIMIC_CXR_FRONTAL_ONLY  MIMIC_CXR_GENERATE_METHOD
#   MAX_EVAL_SAMPLES  MAX_NEW_TOKENS  CONDA_NAME
#
# Slurm 日志：MedEvalKit/llavaradevalresult/slurm_logs/

#SBATCH --job-name=mimic_infer_rrg
#SBATCH --output=llavaradevalresult/slurm_logs/%x_%j.out
#SBATCH --error=llavaradevalresult/slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=L40
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

CONDA_NAME="${CONDA_NAME:-DQllava}"
MEDEVAL_ROOT="${MEDEVAL_ROOT:-${MEDEVALKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
LLAVA_RAD_ROOT="${LLAVA_RAD_ROOT:-${MEDEVAL_ROOT}/LLaVA-Rad}"
RUN_TAG="${RUN_TAG:-mimic_frontal_llavarad_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${MEDEVAL_ROOT}/eval_results/${RUN_TAG}}"
MODEL_NAME="${MODEL_NAME:-LLava_Rad}"
MODEL_PATH="${MODEL_PATH:-${MEDEVAL_ROOT}/DQLLaVAhpc/checkpoints/llavarad051015-8krandom_seed42-val_ap-bs4-ga1-allinone-xca-True-scope-text-alphaTanh-True-3e-1e-4}"
DATASETS_PATH="${DATASETS_PATH:-${MEDEVAL_ROOT}/utils/MIMIC_CXR}"

export MIMIC_CXR_ANNOTATION_JSON="${MIMIC_CXR_ANNOTATION_JSON:-MIMICCXR/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json}"
export MIMIC_CXR_FRONTAL_ONLY="${MIMIC_CXR_FRONTAL_ONLY:-1}"
# 若只想 rule-based 行，设为 rule-based；留空则不过滤 generate_method
export MIMIC_CXR_GENERATE_METHOD="${MIMIC_CXR_GENERATE_METHOD:-}"

export MEDEVALKIT_ROOT="${MEDEVAL_ROOT}"
LLAVA_RAD_WEIGHT_ROOT="${LLAVA_RAD_WEIGHT_ROOT:-${MEDEVAL_ROOT}/LLaVARadWeight}"
export LLAVA_RAD_BASE_MODEL_PATH="${LLAVA_RAD_BASE_MODEL_PATH:-${LLAVA_RAD_WEIGHT_ROOT}/vicuna-7b-v1.5}"

RESULT_ROOT="${RESULT_ROOT:-${MEDEVAL_ROOT}/llavaradevalresult}"
OUT_JSONL="${RESULT_ROOT}/mimic_cxr_preds_${RUN_TAG}_${SLURM_JOB_ID}.jsonl"
EVAL_OUT="${RESULT_ROOT}/rrg_eval_${RUN_TAG}_${SLURM_JOB_ID}"
RUN_NAME="${RUN_NAME:-mimic_frontal_rrg_${SLURM_JOB_ID}}"
BOOTSTRAP_CI="${BOOTSTRAP_CI:-false}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-42}"
USE_VLLM="${USE_VLLM:-False}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_IMAGE_NUM="${MAX_IMAGE_NUM:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.0001}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1}"
REASONING="${REASONING:-False}"
TEST_TIMES="${TEST_TIMES:-1}"
USE_LLM_JUDGE="${USE_LLM_JUDGE:-False}"
GPT_MODEL="${GPT_MODEL:-openai/gpt-4o}"
JUDGE_MODEL_TYPE="${JUDGE_MODEL_TYPE:-openai}"
API_KEY="${API_KEY:-None}"
BASE_URL="${BASE_URL:-None}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"

ENABLE_PCA_VIZ="${ENABLE_PCA_VIZ:-False}"
ENABLE_CKNNA_VIZ="${ENABLE_CKNNA_VIZ:-False}"
PCA_MAX_SAMPLES="${PCA_MAX_SAMPLES:-8}"
PCA_OUTPUT_SUBDIR="${PCA_OUTPUT_SUBDIR:-pca_viz}"
PCA_LAYER_IDS="${PCA_LAYER_IDS:-}"
PCA_VISUAL_SOURCE="${PCA_VISUAL_SOURCE:-global}"
CKNNA_TOPK="${CKNNA_TOPK:-10}"

mkdir -p "${MEDEVAL_ROOT}/llavaradevalresult/slurm_logs" "${OUTPUT_PATH}"

# shellcheck disable=SC1091
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_NAME}"
fi
echo "[mimic_infer_rrg] conda=${CONDA_NAME} python=$(command -v python || true) JOB=${SLURM_JOB_ID:-local}"

export PYTHONPATH="${MEDEVAL_ROOT}:${MEDEVAL_ROOT}/DQLLaVAhpc:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
unset PYTORCH_CUDA_ALLOC_CONF || true

cd "${MEDEVAL_ROOT}"

echo "========== Phase 1: eval.py MIMIC_CXR (frontal) =========="
python eval.py \
  --eval_datasets MIMIC_CXR \
  --datasets_path "${DATASETS_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --seed "${SEED}" \
  --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
  --tensor_parallel_size 1 \
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
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --enable_pca_viz "${ENABLE_PCA_VIZ}" \
  --enable_cknna_viz "${ENABLE_CKNNA_VIZ}" \
  --pca_max_samples "${PCA_MAX_SAMPLES}" \
  --pca_output_subdir "${PCA_OUTPUT_SUBDIR}" \
  --pca_layer_ids "${PCA_LAYER_IDS}" \
  --pca_visual_source "${PCA_VISUAL_SOURCE}" \
  --cknna_topk "${CKNNA_TOPK}"

RESULTS_JSON="${OUTPUT_PATH}/${MODEL_NAME}/MIMIC_CXR/results.json"
if [[ ! -f "${RESULTS_JSON}" ]]; then
  echo "ERROR: Phase 1 did not produce ${RESULTS_JSON}" >&2
  exit 1
fi

echo "========== Phase 2: RRG (run.py) =========="
export RESULTS_JSON
export RESULT_ROOT
export OUT_JSONL
export EVAL_OUT
export RUN_NAME
export BOOTSTRAP_CI
export LLAVA_RAD_ROOT
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

bash "${LLAVA_RAD_ROOT}/scripts/run_rrg_from_results_json.sh"

echo "[mimic_infer_rrg] Done. Infer: ${OUTPUT_PATH}/${MODEL_NAME}/MIMIC_CXR/"
echo "[mimic_infer_rrg] RRG CSV: ${EVAL_OUT}/main.csv"
