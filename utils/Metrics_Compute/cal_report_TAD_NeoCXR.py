#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import json
import os
from collections import defaultdict

# ========== 复用你的归一化函数（务必与训练时一致）==========
def normalize_p(p_str):
    if not isinstance(p_str, str) or not p_str.strip():
        return "其他"
    p = p_str.strip()
    if re.search(r'湿肺|暂时性呼吸急促|TTN', p, re.IGNORECASE):
        return "肺渗出"
    if re.search(r'NRDS|呼吸窘迫综合征|RDS', p, re.IGNORECASE):
        return "NRDS"
    if re.search(r'BPD|支气管肺发育不良', p, re.IGNORECASE):
        return "支气管肺发育不良"
    if re.search(r'气胸', p):
        return "气胸"
    if re.search(r'实变', p):
        return "肺实变"
    if re.search(r'肺炎|肺部感染|肺感染|炎性|炎症', p, re.IGNORECASE):
        return "新生儿肺炎"
    if re.search(r'渗出|模糊影|斑片影|磨玻璃|散在.*?影|片状影|密度增高影', p):
        return "肺渗出"
    p_clean = re.sub(r'左[侧]?|右[侧]?|双侧|肺[野叶]?', '', p)
    if re.search(r'纹理.*?(增多|增粗|稍多|稍粗|略多|略粗|轻度.*?多|轻度.*?粗|紊乱|模糊)', p_clean, re.IGNORECASE):
        return "肺纹理异常"
    # 注意：心影、不张等已归入“其他”
    return "其他"

def normalize_d(d_str):
    if not isinstance(d_str, str) or not d_str.strip():
        return "相仿"
    d = d_str.strip()
    if any(kw in d for kw in ["好转", "吸收", "减少", "缩小", "减低", "改善"]):
        return "improved"
    elif any(kw in d for kw in ["进展", "增多", "恶化", "增加", "加重"]):
        return "progress"
    else:
        return "similar"

def normalize_s(s_str):
    if not isinstance(s_str, str) or not s_str.strip():
        return ""
    s = s_str.strip()
    if "" in s:
        return ""
    elif any(kw in s for kw in ["slightly", "略", "大致", "轻度"]):
        return "mild"
    elif any(kw in s for kw in ["有所", "部分", "中度"]):
        return "moderate"
    elif any(kw in s for kw in ["明显", "基本", "显著", "重度"]):
        return "severe"
    else:
        return ""

# ========== 从模型报告 R 中提取 temporal 三元组 ==========
def extract_temporal_triplets_from_report(report_text):
    """
    从模型生成的完整报告中提取 temporal 三元组列表
    """
    # 1. 提取“影像结论”部分
    if "影像结论" in report_text:
        start = report_text.find("影像结论")
        end = report_text.find("疾病诊断", start)
        if end == -1:
            end = len(report_text)
        conclusion = report_text[start:end].strip()
    else:
        # 若无明确分段，用整段
        conclusion = report_text

    # 2. 分句（按中文句号、分号、换行）
    sentences = re.split(r'[。；;\n]', conclusion)
    triplets = []

    # 3. temporal 关键词正则
    temporal_pattern = re.compile(r'(.*?)(?:较[前次]|较上次|较前片|较之前|与前片|与上次|较前)(.*?)(?:[。；;，,]|$)')

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # 检查是否含 temporal 关键词
        if not re.search(r'较[前次]|较上次|较前片|较之前|与前', sent):
            continue

        # 尝试匹配
        match = temporal_pattern.search(sent)
        if match:
            p_raw = match.group(1).strip()
            d_s_raw = match.group(2).strip()
        else:
            # 若无法分割，整句作为 P，D 默认“相仿”（保守）
            p_raw = sent
            d_s_raw = ""

        # 归一化
        p_norm = normalize_p(p_raw)
        d_norm = normalize_d(d_s_raw)
        s_norm = normalize_s(d_s_raw)  # S 从 D 描述中提取

        triplets.append({"P": p_norm, "D": d_norm, "S": s_norm})

    # 去重（可选）
    unique_triplets = []
    seen = set()
    for tri in triplets:
        key = (tri["P"], tri["D"], tri["S"])
        if key not in seen:
            seen.add(key)
            unique_triplets.append(tri)
    return unique_triplets

# ========== 评估函数 ==========
def evaluate_temporal_triplets(gt_file, pred_reports):
    """
    gt_file: test_temporal.json 路径（含人工标注 temporal_triplets）
    pred_reports: {id: model_generated_report_string}
    """
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    total_samples = 0
    exact_match = 0
    p_correct = d_correct = s_correct = 0
    total_triplets = 0

    for item in gt_data:
        sample_id = item["id"]
        if sample_id not in pred_reports:
            continue
        total_samples += 1

        gt_triplets = item.get("temporal_triplets", [])
        pred_report = pred_reports[sample_id]
        pred_triplets = extract_temporal_triplets_from_report(pred_report)

        # 简单策略：取第一个三元组比较（或做集合匹配）
        # 这里假设每个样本只关注一个主要变化（你可扩展为多三元组匹配）
        if not gt_triplets:
            continue

        gt_tri = gt_triplets[0]  # 取第一个作为主标签
        if pred_triplets:
            pred_tri = pred_triplets[0]
        else:
            pred_tri = {"P": "其他", "D": "相仿", "S": ""}

        # 三元组完全匹配
        if (gt_tri["P"] == pred_tri["P"] and
            gt_tri["D"] == pred_tri["D"] and
            gt_tri["S"] == pred_tri["S"]):
            exact_match += 1

        # 字段级准确率
        if gt_tri["P"] == pred_tri["P"]:
            p_correct += 1
        if gt_tri["D"] == pred_tri["D"]:
            d_correct += 1
        if gt_tri["S"] == pred_tri["S"]:
            s_correct += 1

        total_triplets += 1

    # 计算指标
    results = {
        "样本数": total_samples,
        "三元组完全准确率": exact_match / total_triplets if total_triplets else 0,
        "P 准确率": p_correct / total_triplets if total_triplets else 0,
        "D 准确率": d_correct / total_triplets if total_triplets else 0,
        "S 准确率": s_correct / total_triplets if total_triplets else 0,
    }
    return results

# ========== 使用示例 ==========
def main():
    # 1. 加载模型预测报告（假设你有一个 {id: report} 的 JSON）
    pred_report_path = "/path/to/your/model_predictions.json"  # 格式: {"28": "影像结论：... 疾病诊断：..."}
    with open(pred_report_path, 'r', encoding='utf-8') as f:
        pred_reports = json.load(f)

    # 2. 评估
    gt_file = "/media/hdd2/MedEvalKit/LLaVARad/hospital_data/NeoCXR/splits/test_temporal.json"
    results = evaluate_temporal_triplets(gt_file, pred_reports)

    # 3. 打印
    print("Temporal Triplets 评估结果:")
    for k, v in results.items():
        if "准确率" in k:
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()