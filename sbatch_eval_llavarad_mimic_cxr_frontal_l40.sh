#!/bin/bash
#
# L40 + DQllava + LLava_Rad on MIMIC_CXRfrontal (frontal AP/PA, preprocessed test.json).
#
# Before first run (login node, from MedEvalKit root):
#   cd /share/home/u16050/apps/Lynne/MLLM/MedEvalKit
#   python -m utils.MIMIC_CXRfrontal.preprocess
#   # Optional: MIMIC_CXR_GENERATE_METHOD=rule-based  MIMIC_CXRFRONTAL_PREPROCESS_INPUT=...  MIMIC_CXRFRONTAL_PREPROCESS_OUTPUT=...
#
# test.json = list of LLaVA-Rad chat dicts (conversations, image, id, …). Model prompt = human conversation only.
# cal_metrics / RRG: findings from gpt turn, impression from JSON top-level (keys attached at load).
# Images: export DQLLAVA_MIMIC_JPG_ROOT=... if not using default under utils/MIMIC_CXR/MIMIC_CXR/data/...
#
#SBATCH --job-name=llavarad_mimic_frontal
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=2918425700@qq.com

set -euo pipefail

PROJECT_ROOT="${MEDEVALKIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_NAME="${CONDA_NAME:-DQllava}"

mkdir -p "${PROJECT_ROOT}/slurm_logs"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_NAME}"
else
  source /share/apps/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_NAME}"
fi
_EXPECT_ENV="${MEDEVALKIT_CONDA_ENV:-${CONDA_NAME}}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${_EXPECT_ENV}" ]]; then
  echo "[FATAL] 需要 conda 环境 '${_EXPECT_ENV}'，当前 CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}' python='$(command -v python || true)'" >&2
  exit 1
fi
unset PYTORCH_CUDA_ALLOC_CONF
_PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CUDA_RUNTIME_LIB="${CONDA_PREFIX}/lib/python${_PYVER}/site-packages/nvidia/cuda_runtime/lib"
CUSPARSE_LIB="${CONDA_PREFIX}/lib/python${_PYVER}/site-packages/nvidia/cusparse/lib"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}:${CUSPARSE_LIB}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export USE_VLLM="${USE_VLLM:-False}"
export LLAVA_RAD_USE_UPSTREAM="${LLAVA_RAD_USE_UPSTREAM:-0}"
# 默认只加载官方发布的 LLaVARadWeight（根目录：adapter / projector 等 + 下面 vicuna 底座），不加载 DQLLaVAhpc 自训 checkpoint。
# 若要用自训权重：export MODEL_PATH=/path/to/DQLLaVAhpc/checkpoints/...
_LLAVA_WEIGHT="${PROJECT_ROOT}/LLaVARadWeight"
export MODEL_PATH="${MODEL_PATH:-${_LLAVA_WEIGHT}}"
export LLAVA_RAD_WEIGHT_ROOT="${LLAVA_RAD_WEIGHT_ROOT:-${_LLAVA_WEIGHT}}"
export LLAVA_RAD_BASE_MODEL_PATH="${LLAVA_RAD_BASE_MODEL_PATH:-${LLAVA_RAD_WEIGHT_ROOT}/vicuna-7b-v1.5}"

export OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/eval_results/mimic_cxr_frontal_l40}"
export EVAL_DATASETS="${EVAL_DATASETS:-MIMIC_CXRfrontal}"
export DATASETS_PATH="${DATASETS_PATH:-${PROJECT_ROOT}/utils}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] LLAVA_RAD_BASE_MODEL_PATH=${LLAVA_RAD_BASE_MODEL_PATH}"
echo "[INFO] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[INFO] EVAL_DATASETS=${EVAL_DATASETS}"
echo "[INFO] DATASETS_PATH=${DATASETS_PATH}"

bash "${PROJECT_ROOT}/eval.sh"
