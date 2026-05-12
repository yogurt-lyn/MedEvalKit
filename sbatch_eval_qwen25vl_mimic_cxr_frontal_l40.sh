#!/usr/bin/env bash
#
# L40 + conda「qwencomp」+ Qwen2.5-VL：MIMIC_CXRfrontal（与 sbatch_eval_llavarad_mimic_cxr_frontal_l40_joint_ckpt.sh 同一数据与 eval.sh 流程）。
# 默认加载本地 HF 权重：data/LYN/Qwencomp/Qwen2.5-VL-7B-Instruct。
#
# 提交示例（关闭 PCA/CKNNA，指定输出目录，与 joint_projector 评测对齐）:
#   cd /share/home/u16050/apps/Lynne/MLLM/MedEvalKit
#   sbatch --export=ALL,MAX_EVAL_SAMPLES=0,ENABLE_PCA_VIZ=False,ENABLE_CKNNA_VIZ=False,OUTPUT_PATH=/share/home/u16050/apps/Lynne/MLLM/MedEvalKit/eval_results/mimic_cxr_frontal_qwen25vl_instruct sbatch_eval_qwen25vl_mimic_cxr_frontal_l40.sh
#
# 覆盖权重：export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct 再 sbatch 本脚本。
#
#SBATCH --job-name=qwen25vl_mimic_frontal
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
CONDA_NAME="${CONDA_NAME:-qwencomp}"
export CONDA_NAME
export MEDEVALKIT_ROOT="${MEDEVALKIT_ROOT:-${PROJECT_ROOT}}"

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
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}:${CUSPARSE_LIB}:${CONDA_PREFIX}/lib:/share/apps/cuda-11.8/lib64:/share/apps/cuda-11.8/targets/x86_64-linux/lib:/share/apps/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"
export NLTK_DATA="${NLTK_DATA:-${PROJECT_ROOT}/nltk_data}"

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export USE_VLLM="${USE_VLLM:-False}"

export MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/Qwen2.5-VL-7B-Instruct}"

export OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/eval_results/mimic_cxr_frontal_qwen25vl_instruct}"
export EVAL_DATASETS="${EVAL_DATASETS:-MIMIC_CXRfrontal}"
export DATASETS_PATH="${DATASETS_PATH:-${PROJECT_ROOT}/utils}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[INFO] EVAL_DATASETS=${EVAL_DATASETS}"
echo "[INFO] DATASETS_PATH=${DATASETS_PATH}"
echo "[INFO] ENABLE_PCA_VIZ=${ENABLE_PCA_VIZ:-}"
echo "[INFO] ENABLE_CKNNA_VIZ=${ENABLE_CKNNA_VIZ:-}"
echo "[INFO] MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-}"

bash "${PROJECT_ROOT}/eval.sh"
