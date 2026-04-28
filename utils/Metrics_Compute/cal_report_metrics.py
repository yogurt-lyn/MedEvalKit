import nltk
nltk.data.path.append("/data_lyn/MedEvalKit/utils/Metrics_Compute/nltk_data")
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
import sys
sys.setrecursionlimit(10000)
from rouge import Rouge

import re
import unicodedata

# from cider.cidereval import cider, ciderD
from bert_score import BERTScorer
from RaTEScore import RaTEScore
# from RaTEScore.RaTEScore.scorer import RaTEScore
# from green_score import GREEN
from tqdm import tqdm
import json

import os
import nltk

import numpy as np


# caculate the metrics of ratescore, green score, cider, bleu, rouge, meteor
# other metrics please refer to https://github.com/rajpurkarlab/CXR-Report-Metric

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['PYTHONIOENCODING'] = 'utf-8'

# /media/hdd2/MedEvalKit/eval_results/{}/LLava_Rad/LLava_Rad/IU_XRAY
# IU_XRAY,CheXpert_Plus,MIMIC_CXR /media/hdd2/MedEvalKit/eval_results/{}/Huatuo/MIMIC_CXR
# model_name = "/data_lyn/MedEvalKit/eval_results/B_mix1r=4align_loss0.5-lora-1e-1e-4/NeoRed"#my2861: /media/hdd2/MedEvalKit/eval_results/{}
# datasets = ["IU_XRAY"]#["IU_XRAY","CheXpert_Plus","MIMIC_CXR"]

model_name = "/data_lyn/MedEvalKit/eval_results/monthb163b32r641e-4baselinew3.4allloss-lora-1e-1e-4/NeoRed"
datasets = ["MIMIC_CXR"]


# /media/hdd2/MedEvalKit/eval_results/LLava_Rad_NeoCXR_d0_1.6k/LLava_Rad/NeoCXR
ratescore = RaTEScore(bert_model = "/data_lyn/MedEvalKit/utils/Metrics_Compute/RaTE-NER-Deberta",eval_model='/data_lyn/MedEvalKit/utils/Metrics_Compute/BioLORD-2023-C')
# green_scorer = GREEN("GREEN-radllama2-7b", output_dir=".")


# BioClinicalBERT Local Model Path
bio_clinical_bert_path = "/data_lyn/MedEvalKit/models/Bio_ClinicalBERT"

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

def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]

for dataset in datasets:
        results_path = f"{model_name}/{dataset}/results.json"
        output_metrics_path = f"{model_name}/{dataset}/metrics_2.json"
        if not os.path.exists(results_path):
                print(f"Results file {results_path} does not exist.")
                continue

        with open(results_path,"r", encoding='utf-8') as f:
                datas = json.load(f)

        reports = []
        reports2 = []
        preds = []

        total_bleu1 = 0
        total_bleu2 = 0
        total_bleu3 = 0
        total_bleu4 = 0

        total_rouge1 = 0
        total_rouge2 = 0
        total_rougel = 0

        total_precision = 0
        total_recall = 0
        total_f1 = 0

        total_meteor_scores = 0

        messages_list = []
        skip=0
        
        for sample in tqdm(datas):
                
                response = sample["response"]
                # print(response)
                # if response == "":
                #         continue
                if (not isinstance(response, str) or response.strip() == "" or response.strip() == "."):
                        skip+=1
                        continue
                findings = sample["findings"]
                impression = sample["impression"]
                golden = f"Findings: {findings} Impression: {impression}."


                tokenized_response =prep_reports([response.lower()])[0]
                tokenized_golden = prep_reports([golden.lower()])[0] 

                reports.append(golden)
                reports2.append([golden.lower()])

                preds.append(response)

                bleu1 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1])
                bleu2 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.5,0.5])
                bleu3 = sentence_bleu([tokenized_golden], tokenized_response, weights=[1/3,1/3,1/3])
                bleu4 = sentence_bleu([tokenized_golden], tokenized_response, weights=[0.25,0.25,0.25,0.25])
                total_bleu1 += bleu1
                total_bleu2 += bleu2
                total_bleu3 += bleu3
                total_bleu4 += bleu4
                

                rouge_scores = rouge_scorer.get_scores(response.lower(), golden.lower())
                total_rouge1 += rouge_scores[0]["rouge-1"]["f"]
                total_rouge2 += rouge_scores[0]["rouge-2"]["f"]
                total_rougel += rouge_scores[0]["rouge-l"]["f"]

                meteor_scores = single_meteor_score(hypothesis = tokenized_response,reference  = tokenized_golden)
                total_meteor_scores += meteor_scores

        # print("begin to compute cider metrics...")
        # cider_score = cider(predictions=[pred.lower() for pred in preds], references= reports2)
        # cider_score = cider_score["avg_score"]
        print("skipnum:",skip)
        print("begin to compute RaTE score...")
        rate_scores = ratescore.compute_score(preds, reports)
        rate_score = sum(rate_scores)/len(rate_scores)

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

        # print("begin to compute Green score...")
        # green_mean, green_std, green_score_list, summary, result_df = green_scorer(reports, preds)


        print(f"Meteor Score: {total_meteor_scores/len(datas)}")
        print(f"RaTE Score: {rate_score}")
        # print(f"Green Score: {green_mean} ± {green_std}")
        # print(f"Cider Score: {cider_score}")

        print("Metrics computed successfully!")
        print(f"Bleu-1: {total_bleu1/len(datas)}")
        print(f"Bleu-2: {total_bleu2/len(datas)}")
        print(f"Bleu-3: {total_bleu3/len(datas)}")
        print(f"Bleu-4: {total_bleu4/len(datas)}")
        print(f"Rouge-1: {total_rouge1/len(datas)}")
        print(f"Rouge-2: {total_rouge2/len(datas)}")
        print(f"Rouge-L: {total_rougel/len(datas)}")
        print(f"BERTScore (Precision): {bertscore_precision:.4f}")
        print(f"BERTScore (Recall): {bertscore_recall:.4f}")
        print(f"BERTScore (F1): {bertscore_f1:.4f}")

        metrics = {
                # "cider": float(cider_score),
                "meteor": float(total_meteor_scores/len(datas)),
                "bleu1": float(total_bleu1/len(datas)),
                "bleu2": float(total_bleu2/len(datas)),
                "bleu3": float(total_bleu3/len(datas)),
                "bleu4": float(total_bleu4/len(datas)),
                "rouge1": float(total_rouge1/len(datas)),
                "rouge2": float(total_rouge2/len(datas)),
                "rougeL": float(total_rougel/len(datas)),
                "rate": float(rate_score),
                "bertscore_f1": float(bertscore_f1),
                "bertscore_precision": float(bertscore_precision),
                "bertscore_recall": float(bertscore_recall),
                # "green_mean": float(green_mean),
                # "green_std": float(green_std),
                # "green_score" : f"{green_mean} ± {green_std}",
        }

        with open(output_metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)