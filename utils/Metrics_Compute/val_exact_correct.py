from ExactTriple import extract_triplets

import json
import os
import re
from typing import List, Dict

def main():
    input_path = "/media/hdd2/MedEvalKit/LLaVARad/hospital_data/NeoCXR/triple_splits/alldata_tosplit.json"
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct_TDA = []  # 存储每个样本的 TDA 分数
    all_P_acc=[]
    all_D_acc=[]
    all_S_acc=[]

    for sample in data:
        if sample.get("data_type") != "Temporal":
            continue

        # 获取模型生成的英文报告（GT 报告）
        conv_en = sample.get("conversation_en", [])
        if len(conv_en) < 2:
            continue
        response = conv_en[1].get("value", "")
        if not response.strip():
            continue

        # === 提取预测三元组 ===
        pred_PDS_list= extract_triplets(response)
        pred_PDS = pred_PDS_list[0] if pred_PDS_list else {"P": "Other", "D": "similar", "S": ""}

        # === 获取 GT 三元组 ===
        gt_PDS_list = sample.get("temporal_triplets_en", [])
        if not gt_PDS_list:
            gt_PDS = {"P": "Other", "D": "Similar", "S": ""}
        else:
            gt_PDS = gt_PDS_list[0]  # 取第一个三元组

        # === 计算单样本 P/D/S 准确率 ===
        # P_acc = 1.0 if pred_PDS["P"] == gt_PDS["P"] else 0.0
        # D_acc = 1.0 if pred_PDS["D"] == gt_PDS["D"] else 0.0
        # S_acc = 1.0 if pred_PDS["S"] == gt_PDS["S"] else 0.0

        # === 计算单样本 P/D/S 准确率（统一转小写后比较）===
        P_acc = 1.0 if pred_PDS["P"].lower() == gt_PDS["P"].lower() else 0.0
        D_acc = 1.0 if pred_PDS["D"].lower() == gt_PDS["D"].lower() else 0.0
        S_acc = 1.0 if pred_PDS["S"].lower() == gt_PDS["S"].lower() else 0.0

        # if P_acc == 0.0 or D_acc == 0.0:
        #     print("\n" + "="*60)
        #     print("P 或 D 预测错误")
        #     print(f"模型报告: {response.strip()}")
        #     print(f"预测 P: '{pred_PDS['P']}', GT P: '{gt_PDS['P']}'")
        #     print(f"预测 D: '{pred_PDS['D']}', GT D: '{gt_PDS['D']}'")
        #     print(f"预测 S: '{pred_PDS['S']}', GT S: '{gt_PDS['S']}'")

        # if D_acc == 0.0:
        #     print("\n" + "="*60)
        #     print("D 预测错误")
        #     print(f"模型报告: {response.strip()}")

        #     print(f"预测 D: '{pred_PDS['D']}', GT D: '{gt_PDS['D']}'")


        if P_acc == 0.0:
            print("\n" + "="*60)
            print("P 预测错误")
            print(f"模型报告: {response.strip()}")

            # print("P_text:",P_text)
            print(f"预测 P: '{pred_PDS['P']}', GT P: '{gt_PDS['P']}'")

        # if S_acc == 0.0:
        #     print("\n" + "="*60)
        #     print("S 预测错误")
        #     print(f"模型报告: {response.strip()}")

        #     print(f"预测 S: '{pred_PDS['S']}', GT S: '{gt_PDS['S']}'")
        #     # print(f"预测 P: '{pred_PDS['P']}', GT P: '{gt_PDS['P']}'")
        #     # print(f"预测 D: '{pred_PDS['D']}', GT D: '{gt_PDS['D']}'")
  
  
  

        # === 计算该样本的 TDA ===
        sample_TDA = 0.5 * P_acc + 0.3 * D_acc + 0.2 * S_acc
        correct_TDA.append(sample_TDA)
        all_P_acc.append(P_acc)
        all_D_acc.append(D_acc)
        all_S_acc.append(S_acc)
    # === 输出结果 ===
    if not correct_TDA:
        print("未找到有效的 Temporal 样本")
        return

    avg_TDA = sum(correct_TDA) / len(correct_TDA)
    print(f"验证完成！")
    print(f"样本总数: {len(correct_TDA)}")
    print(f"平均 TDA (GT 提取准确率): {avg_TDA:.4f}")

    avg_P = sum(all_P_acc) / len(all_P_acc)
    avg_D = sum(all_D_acc) / len(all_D_acc)
    avg_S = sum(all_S_acc) / len(all_S_acc)

    print(f"P 准确率 (0.5 权重): {avg_P:.4f}")
    print(f"D 准确率 (0.3 权重): {avg_D:.4f}")
    print(f"S 准确率 (0.2 权重): {avg_S:.4f}")

if __name__ == "__main__":
    main()