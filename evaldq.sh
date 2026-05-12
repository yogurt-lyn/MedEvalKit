#!/bin/bash
export PYTHONPATH="/data_lyn/MedEvalKit/DQ_LLaVA:$PYTHONPATH"
# export HF_ENDPOINT=https://hf-mirror.com
# NeoCXR,MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA,Radrestruct
EVAL_DATASETS="IU_XRAY" 
DATASETS_PATH="/data_lyn/MedEvalKit/utils/IU_XRAY" #/media/hdd2/MedEvalKit/LLaVARad/hospital_data  /media/hdd2/MedEvalKit/LLaVARad/hospital_data /media/hdd2/MedEvalKit/LLaVARad/hospital_data/NeoCXR /media/hdd2/xhrdata/datasets /media/hdd2/MedEvalKit/utils/IU_XRAY /media/hdd2/MedEvalKit/utils/VQA_RAD

OUTPUT_PATH="/data_lyn/MedEvalKit/eval_results/IU-XRAY/DQLavaonly282heart" #/{}oneimage
# NeoCXR,TestModel,Qwen2-VL,Qwen2.5-VL,BiMediX2,DQLLava,LLava_Med,LLava_Rad,Huatuo,InternVL,Llama-3.2,LLava,Janus,HealthGPT,BiomedGPT,Vllm_Text,MedGemma,Med_Flamingo,MedDr
MODEL_NAME="DQLLava"
# MODEL_PATH="/media/hdd2/LLaVA-Med/llava-med-v1.5-mistral-7b"
# MODEL_PATH="/media/hdd2/llava-med-v1.5-mistral-7b" #erro：AssertionError: Padding_idx must be within num_embeddings
# MODEL_PATH="/media/hdd2/Qwen2.5-VL-7B-Instruct"
# MODEL_PATH="/media/hdd2/InternVL2_5-8B"
# MODEL_PATH="/media/hdd2/MedEvalKit/LLaVARadWeight" # /media/hdd2/MedEvalKit/LLaVARadWeight /media/hdd2/checkpoints/NeoCXR_baseline_traindatanum_4395-lora-3e-1e-4   /media/hdd2/MedEvalKit/LLaVARad/checkpoints/Temporal_NeoCXR-lora-3e-1e-4
# NeoRad
MODEL_PATH="/data_lyn/DQ_LLaVA/checkpoints/b1visfonly11-lora-3e-1e-4"
# MODEL_PATH="/media/hdd2/MedEvalKit/LLaVARad/checkpoints/Temporal_NeoCXR-lora-3e-1e-4"
# MODEL_PATH="/media/hdd2/llava-1.5-7b-hf"
# MODEL_PATH="/media/hdd2/HuatuoGPT-Vision-7B"

#vllm setting
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="2"
USE_VLLM="True"

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1

# Eval LLM setting
MAX_NEW_TOKENS=8192 #原始8192 公开 2048 Neo 1024
MAX_IMAGE_NUM=2
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="False"
# gpt api model name
GPT_MODEL="openai/gpt-4o" #openrouter:openai/gpt-4o  DeepSeek-R1
JUDGE_MODEL_TYPE="openai"  # openai deepseek or gemini or deepseek or claude openrouter
API_KEY="${API_KEY:-None}"
BASE_URL="https://openrouter.ai/api/v1" #  https://openrouter.ai/api/v1  https://openrouter.ai/api/v1/chat/completions https://api.catflows.com


# #deepseek
# GPT_MODEL="DeepSeek-R1" #openrouter:openai/gpt-4o  DeepSeek-R1
# JUDGE_MODEL_TYPE="deepseek"  # openai deepseek or gemini or deepseek or claude openrouter
# API_KEY="${API_KEY:-None}"
# BASE_URL="https://llmapi.tongji.edu.cn/v1" #  https://openrouter.ai/api/v1  https://openrouter.ai/api/v1/chat/completions https://api.catflows.com


# pass hyperparameters and run python sccript
python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$GPT_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES"
