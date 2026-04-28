
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用GPU 0

import nltk
nltk.data.path.append("/data01/kjt_data/MLLM/MedEvalKit/utils/Metrics_Compute/nltk_data")
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
import sys
sys.setrecursionlimit(10000)
from rouge import Rouge
from nltk.tokenize import word_tokenize


import re
import json
import os
import numpy as np
from tqdm import tqdm

from RaTEScore import RaTEScore
from bert_score import BERTScorer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 模型和数据路径
model_name = "/data01/kjt_data/MLLM/MedEvalKit/eval_results/clinccontextagaintestwoclinic/NeoRed"
datasets = ["NeoCXR"]

# 初始化评分器
ratescore = RaTEScore(
    bert_model="/data01/kjt_data/MLLM/MedEvalKit/utils/Metrics_Compute/RaTE-NER-Deberta",
    eval_model='/data01/kjt_data/MLLM/MedEvalKit/utils/Metrics_Compute/BioLORD-2023-C'
)

# BioClinicalBERT Local Model Path
bio_clinical_bert_path = "/data01/kjt_data/MLLM/MedEvalKit/models/Bio_ClinicalBERT"


# Initialize BERTScorer (using BioClinicalBERT)
print("Loading BERTScorer with BioClinicalBERT...")
bertscorer = BERTScorer(
    model_type=bio_clinical_bert_path,
    num_layers=12,
    rescale_with_baseline=False,
)
print("BERTScorer loaded successfully.")

# Ensure tokenizer max length is bounded to prevent truncation overflow
try:
    bertscorer._tokenizer.model_max_length = 512
    if hasattr(bertscorer._tokenizer, "init_kwargs"):
        bertscorer._tokenizer.init_kwargs["model_max_length"] = 512
except Exception:
    pass

rouge_scorer = Rouge()

def extract_diagnosis(text):
    """提取诊断文本"""
    match = re.search(r'Disease diagnosis[:：]?\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if match:
        diagnosis = match.group(1).strip()
    else:
        diagnosis = text.strip()
    
    # 统一清洗
    diagnosis = re.sub(r'[，,、\s]+', ' ', diagnosis)
    diagnosis = re.sub(r'[。.；;]+$', '', diagnosis)
    return diagnosis.strip().lower()

def unified_tokenizer(text):
    # 统一转化为小写并分词
    tokens = word_tokenize(text.lower())
    # 过滤掉纯标点符号
    tokens = [t for t in tokens if t.isalnum() or t in ['.', ',']]
    return tokens

import re

# def calculate_case_level_specificity(y_true, y_pred):
#     """
#     Case-level Specificity:
#     在 GT 完全无病的样本中，被正确预测为无病的比例
#     """
#     # y_true, y_pred: shape = [N, num_labels]

#     # GT 为完全正常
#     gt_normal = (y_true.sum(axis=1) == 0)

#     # 预测为完全正常
#     pred_normal = (y_pred.sum(axis=1) == 0)

#     # 避免除 0
#     total_normal = gt_normal.sum()
#     if total_normal == 0:
#         return 0.0

#     tn = np.sum(gt_normal & pred_normal)      # 正常 → 正常
#     fp = np.sum(gt_normal & (~pred_normal))   # 正常 → 误诊

#     return tn / (tn + fp + 1e-8)


def calculate_label_level_specificity(y_true, y_pred):
    tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    spec = np.where(tn + fp > 0, tn / (tn + fp), 1.0)
    return spec.mean()  # macro average


def parse_labels_from_text(text, disease_labels):
    """
    增强版标签提取：引入上下文否定词过滤逻辑
    """
    labels = set()
    if not text or not isinstance(text, str):
        return labels

    # 预处理：转小写，规范化空格
    text_lower = " ".join(text.lower().split())

    # 1. 定义否定词集合 (Negation Triggers)
    # 包含前向否定和排除性描述
    negation_words = {
        "no", "not", "none", "negative", "without", "denies", "denied",
        "absence of", "free of", "excluded", "rule out", "ruled out",
        "clear of", "no evidence of", "not seen", "not observed"
    }

    # disease_mappings = {
    #     "Neonatal Respiratory Distress Syndrome (NRDS)": ["respiratory distress syndrome", "nrds", "rds"],
    #     "Neonatal Pneumonia": ["pneumonia"],
    #     "Neonatal Transient Tachypnea (TTN)": ["transient tachypnea", "ttn", "tachypnea"],
    #     "Pneumothorax": ["pneumothorax"],
    #     "Bronchopulmonary Dysplasia(BPD)": ["lung dysplasia", "bronchopulmonary dysplasia", "dysplasia", "hypoplasia"],
    #     # "Fracture": ["fracture"],
    #     "Pleural Effusion": ["pleural effusion", "effusion"],
    #     "Atelectasis": ["atelectasis"],
    #     # "No obvious abnormalities":["no obvious abnormalities","normal"]
    # }

    disease_mappings = {
        "Neonatal Respiratory Distress Syndrome (NRDS)": ["respiratory distress syndrome", "nrds", "rds"],
        "Neonatal Pneumonia": ["pneumonia"],
        "Neonatal Transient Tachypnea (TTN)": ["transient tachypnea", "ttn"],
        "Pneumothorax": ["pneumothorax"],
        "Bronchopulmonary Dysplasia(BPD)": ["lung dysplasia", "bronchopulmonary dysplasia"],
        # "Fracture": ["fracture"],
        "Pleural Effusion": ["pleural effusion", "effusion"],
        "Atelectasis": ["atelectasis"]
    }

    for disease, keywords in disease_mappings.items():
        if disease not in disease_labels:
            continue

        # if disease == "No obvious abnormalities":
        #     for kw in keywords:
        #         if kw in text_lower:
        #             labels.add(disease)
        #             break  # 找到一个即可
        #     continue  # ⬅️ 跳过下面的否定词检测！
        
        for kw in keywords:
            pattern = re.compile(r'\b' + re.escape(kw) + r'\b')
            matches = list(pattern.finditer(text_lower))
            
            for match in matches:
                start_idx = match.start()

                look_back_window = text_lower[max(0, start_idx - 30):start_idx]
                
                is_negated = False
                for neg in negation_words:
                    
                    neg_pattern = re.compile(r'\b' + re.escape(neg) + r'\b')
                    if neg_pattern.search(look_back_window):
                        is_negated = True
                        break
                
                if not is_negated:
                    labels.add(disease)
                    break 

    return labels

# def calculate_weighted_specificity(y_true, y_pred):
#     """计算加权特异性"""
#     tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
#     fp = ((1 - y_true) * y_pred).sum(axis=0)
#     specificity_per_label = np.where(tn + fp > 0, tn / (tn + fp), 1.0)
#     return specificity_per_label.mean()

def calculate_specificity_per_label(y_true, y_pred):
    """
    计算每个标签的 specificity: TN / (TN + FP)
    y_true, y_pred: [N, num_labels]
    返回: [num_labels,] 的 specificity 数组
    """
    tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)  # 真阴性
    fp = ((1 - y_true) * y_pred).sum(axis=0)        # 假阳性
    # 避免除零：若 TN+FP=0（即无阴性样本），spec 设为 1.0（无假阳性即完美）
    specificity_per_label = np.where(tn + fp > 0, tn / (tn + fp), 1.0)
    return specificity_per_label



for dataset in datasets:
    results_path = f"{model_name}/{dataset}/results.json"
    output_metrics_path = f"{model_name}/{dataset}/metrics_clinic.json"
    
    if not os.path.exists(results_path):
        print(f"Results file {results_path} does not exist.")
        continue

    with open(results_path, "r", encoding='utf-8') as f:
        datas = json.load(f)

    # 初始化存储列表
    reports = []
    preds = []
    
    # 用于诊断指标计算的列表
    all_gt_vectors = []
    all_pred_vectors = []
    
    # 其他指标的累加器
    total_bleu1 = total_bleu2 = total_bleu3 = total_bleu4 = 0
    total_rouge1 = total_rouge2 = total_rougel = 0
    total_meteor_scores = 0

    # 样本计数
    total_diagnosis_samples = 0
    skip = 0
    
    # 定义疾病标签
    DISEASE_LABELS = [
        "Neonatal Respiratory Distress Syndrome (NRDS)",
        "Neonatal Pneumonia",
        "Neonatal Transient Tachypnea (TTN)",
        "Pneumothorax",
        "Bronchopulmonary Dysplasia(BPD)",
        # "Fracture",
        "Pleural Effusion",
        "Atelectasis"
        # "No obvious abnormalities"
    ]
    correct_ids = []
    wrong_ids = []
    # 主处理循环
    for sample in tqdm(datas, desc=f"Processing {dataset}"):
        response = sample.get("response", "")
        

        if not isinstance(response, str) or response.strip() == "" or response.strip() == ".":
            skip += 1
            continue
        
        golden = f"Imaging conclusion: {sample['imaging_conclusion_en']} Disease diagnosis: {sample['disease_diagnosis_en']}"

        pred_diagnosis_str = extract_diagnosis(response)
        gt_diagnosis_str = sample["disease_diagnosis_en"]

        gt_labels = parse_labels_from_text(gt_diagnosis_str, DISEASE_LABELS)
        pred_labels = parse_labels_from_text(pred_diagnosis_str, DISEASE_LABELS)

        gt_vec = [1 if label in gt_labels else 0 for label in DISEASE_LABELS]
        pred_vec = [1 if label in pred_labels else 0 for label in DISEASE_LABELS]

        sample_id = sample.get("id", None)  # 如果你的json里字段叫别的就改成对应key

        if sample_id is not None:
            if gt_vec == pred_vec:
                correct_ids.append({
                    "id": sample_id
                })
            else:
                wrong_ids.append({
                    "id": sample_id
                })

        all_gt_vectors.append(gt_vec)
        all_pred_vectors.append(pred_vec)

    
        total_diagnosis_samples += 1
        
        # 3. 计算文本相似度指标
        tokenized_response = unified_tokenizer(response)
        tokenized_golden = unified_tokenizer(golden)
        
        reports.append(golden)
        preds.append(response)
        
        # BLEU分数
        bleu1 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1])
        bleu2 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.5, 0.5])
        bleu3 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1/3, 1/3, 1/3])
        bleu4 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.25, 0.25, 0.25, 0.25])
        
        total_bleu1 += bleu1
        total_bleu2 += bleu2
        total_bleu3 += bleu3
        total_bleu4 += bleu4
        
        # ROUGE分数
        rouge_scores = rouge_scorer.get_scores(response.lower(), golden.lower())
        total_rouge1 += rouge_scores[0]["rouge-1"]["f"]
        total_rouge2 += rouge_scores[0]["rouge-2"]["f"]
        total_rougel += rouge_scores[0]["rouge-l"]["f"]
        
        # METEOR分数
        meteor_scores = single_meteor_score(hypothesis=tokenized_response, reference=tokenized_golden)
        total_meteor_scores += meteor_scores

    print(f"Processed {len(datas)} samples, skipped {skip} invalid responses")
    
    # 检查是否有有效数据
    if total_diagnosis_samples == 0:
        print(f"No valid samples found for dataset {dataset}")
        continue
    
    # 计算文本相似度指标的平均值
    valid_samples = len(datas) - skip
    avg_bleu1 = total_bleu1 / valid_samples
    avg_bleu2 = total_bleu2 / valid_samples
    avg_bleu3 = total_bleu3 / valid_samples
    avg_bleu4 = total_bleu4 / valid_samples
    avg_rouge1 = total_rouge1 / valid_samples
    avg_rouge2 = total_rouge2 / valid_samples
    avg_rougel = total_rougel / valid_samples
    avg_meteor = total_meteor_scores / valid_samples
    
    if len(all_gt_vectors) > 0:
        y_true = np.array(all_gt_vectors)
        y_pred = np.array(all_pred_vectors)

        from sklearn.metrics import f1_score, precision_score, recall_score

        # Exact match（样本级）
        exact_match = (y_true == y_pred).all(axis=1).mean()

        # Label-level accuracy
        label_accuracies = (y_true == y_pred).mean(axis=0)
        label_level_accuracy = label_accuracies.mean()

        # F1 / Precision / Recall
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)

        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # # Case-level Specificity（关键）
        # case_specificity = calculate_case_level_specificity(y_true, y_pred)

        # 计算 per-label specificity
        spec_per_label = calculate_specificity_per_label(y_true, y_pred)

        # --- Macro Specificity: 所有疾病的 specificity 简单平均 ---
        specificity_macro = spec_per_label.mean()

        # --- Weighted Specificity: 按每个疾病的 **真实阴性样本数** 加权平均 ---
        tn_per_label = ((1 - y_true) * (1 - y_pred)).sum(axis=0)  # 各标签 TN 数
        fp_per_label = ((1 - y_true) * y_pred).sum(axis=0)        # 各标签 FP 数
        weights = tn_per_label + fp_per_label  # 即真实阴性样本数（TN + FP）
        if weights.sum() > 0:
            specificity_weighted = np.average(spec_per_label, weights=weights)
        else:
            specificity_weighted = 1.0

        # --- Micro Specificity: 全局 TN / (全局 TN + 全局 FP) ---
        tn_total = ((1 - y_true) * (1 - y_pred)).sum()
        fp_total = ((1 - y_true) * y_pred).sum()
        specificity_micro = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 1.0

    else:
        exact_match = label_level_accuracy = 0.0
        f1_macro = f1_micro = f1_weighted = 0.0
        precision_macro = recall_macro =specificity_macro = 0.0
        case_specificity = 0.0
        precision_weighted = recall_weighted = specificity_weighted = 0.0
        precision_micro = recall_micro = specificity_micro = 0.0

    # 计算RaTE分数
    print("Computing RaTE score...")
    rate_scores = ratescore.compute_score(preds, reports)
    rate_score = sum(rate_scores) / len(rate_scores) if rate_scores else 0.0

    # 计算BERTScore分数
    print("Computing BERTScore with BioClinicalBERT...")
    if preds and reports:
        P, R, F1 = bertscorer.score(preds, reports)
        bertscore_precision = P.mean().item()
        bertscore_recall = R.mean().item()
        bertscore_f1 = F1.mean().item()
    else:
        bertscore_precision = 0.0
        bertscore_recall = 0.0
        bertscore_f1 = 0.0
    
    # 打印结果
    print(f"\n=== Results for {dataset} ===")
    print(f"Total samples: {len(datas)}")
    print(f"Valid samples: {valid_samples}")
    print(f"Skipped: {skip}")
    print(f"\nText Similarity Metrics:")
    print(f"Meteor Score: {avg_meteor:.4f}")
    print(f"BLEU-1: {avg_bleu1:.4f}")
    print(f"BLEU-2: {avg_bleu2:.4f}")
    print(f"BLEU-3: {avg_bleu3:.4f}")
    print(f"BLEU-4: {avg_bleu4:.4f}")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougel:.4f}")
    print(f"RaTE Score: {rate_score:.4f}")
    print(f"BERTScore (Precision): {bertscore_precision:.4f}")
    print(f"BERTScore (Recall): {bertscore_recall:.4f}")
    print(f"BERTScore (F1): {bertscore_f1:.4f}")
    
    print(f"\nClinical Metrics:")
    print(f"Diagnosis Exact Match Accuracy: {exact_match:.4f}")
    print(f"Diagnosis Label-level Accuracy: {label_level_accuracy:.4f}")

    print(f"Diagnosis F1 (macro): {f1_macro:.4f}")
    print(f"Diagnosis Precision (macro): {precision_macro:.4f}")
    print(f"Diagnosis Recall/Sensitivity (macro): {recall_macro:.4f}")
    print(f"Diagnosis specificity_macro): {specificity_macro:.4f}")

    print(f"Diagnosis F1 (micro): {f1_micro:.4f}")
    print(f"Diagnosis Precision (micro): {precision_micro:.4f}")
    print(f"Diagnosis Recall/Sensitivity (micro): {recall_micro:.4f}")
    print(f"Diagnosis specificity_micro): {specificity_micro:.4f}")
    
    print(f"Diagnosis F1 (weighted): {f1_weighted:.4f}")
    print(f"Diagnosis Precision (weighted): {precision_weighted:.4f}")
    print(f"Diagnosis Recall/Sensitivity (weighted): {recall_weighted:.4f}")
    print(f"Diagnosis specificity_weighted): {specificity_weighted:.4f}")
    
    
    # 保存指标到文件
    metrics = {
        "meteor": float(avg_meteor),
        "bleu1": float(avg_bleu1),
        "bleu2": float(avg_bleu2),
        "bleu3": float(avg_bleu3),
        "bleu4": float(avg_bleu4),
        "rouge1": float(avg_rouge1),
        "rouge2": float(avg_rouge2),
        "rougeL": float(avg_rougel),
        "rate": float(rate_score),
        "bertscore_f1": float(bertscore_f1),
        "bertscore_precision": float(bertscore_precision),
        "bertscore_recall": float(bertscore_recall),
        "diagnosis_exact_match_accuracy": float(exact_match),
        "diagnosis_label_level_accuracy": float(label_level_accuracy),
        "diagnosis_f1_macro": float(f1_macro),
        "diagnosis_precision_macro": float(precision_macro),
        "diagnosis_recall/sensitivity_macro": float(recall_macro),
        "diagnosis_specificity_macro": float(specificity_macro),
        "diagnosis_f1_micro": float(f1_micro),
        "diagnosis_precision_micro": float(precision_micro),
        "diagnosis_recall_micro": float(recall_micro),
        "diagnosis_specificity_micro": float(specificity_micro),
        "diagnosis_f1_weighted": float(f1_weighted),
        "diagnosis_precision_weighted": float(precision_weighted),
        "diagnosis_recall/sensitivity_weighted": float(recall_weighted),
        "diagnosis_specificity_weighted": float(specificity_weighted),
        "sample_statistics": {
            "total_samples": len(datas),
            "valid_samples": valid_samples,
            "skipped_samples": skip
        }
    }
    
    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    correct_path = f"{model_name}/{dataset}/correct_ids.json"
    wrong_path = f"{model_name}/{dataset}/wrong_ids.json"

    with open(correct_path, "w") as f:
        json.dump(correct_ids, f, indent=4)

    with open(wrong_path, "w") as f:
        json.dump(wrong_ids, f, indent=4)

    print(f"Correct IDs saved to: {correct_path}")
    print(f"Wrong IDs saved to: {wrong_path}")
    
    print(f"\nMetrics saved to: {output_metrics_path}")


    