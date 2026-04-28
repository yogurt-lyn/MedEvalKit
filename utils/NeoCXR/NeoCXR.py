import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

import numpy as np

from ..utils import save_json,extract
from ..base_dataset import BaseDataset

class NeoCXR(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

        # prompt_single_path="/data_lyn/MedEvalKit/utils/NeoCXR/prompt/single.txt"
        # prompt_temporal_path="/data_lyn/MedEvalKit/utils/NeoCXR/prompt/temporal.txt"

        # with open(prompt_single_path, 'r', encoding='utf-8') as f:
        #     self.prompt_single = f.read().strip()
        
        # with open(prompt_temporal_path, 'r', encoding='utf-8') as f:
        #     self.prompt_temporal = f.read().strip()
        
        # prompt_clinic_path="/data_lyn/MedEvalKit/utils/NeoCXR/prompt/clinic_prompt.txt"
        prompt_clinic_path="/data01/kjt_data/MLLM/MedEvalKit/utils/NeoCXR/prompt/clinic_prompt_token.txt"

        with open(prompt_clinic_path, 'r', encoding='utf-8') as f:
            self.prompt_clinic = f.read().strip()
    
    # def build_prompt(self, sample):
    #     clinical_info = sample["clinical_information_en"].strip()

    #     if sample["data_type"] == "Single":
    #         prompt = self.prompt_single
    #     elif sample["data_type"] == "Temporal":
    #         prompt = self.prompt_temporal
    #     else:
    #         raise ValueError(f"Unknown data_type: {sample['data_type']}")

    #     if "__CLINICAL_INFO__" not in prompt:
    #         raise ValueError("Prompt template missing __CLINICAL_INFO__ placeholder")

    #     return prompt.replace("__CLINICAL_INFO__", clinical_info)

    def build_prompt(self, sample):

        image_field = sample.get("image", "").strip()
        if not image_field:
            raise ValueError("Sample missing image field")

        image_paths = [p.strip() for p in image_field.split(",") if p.strip()]
        if len(image_paths) == 0:
            raise ValueError("No valid image paths found in image field")

        image_tokens = "".join(["<image>"] * len(image_paths))


        clinical_structured = sample.get("clinical_structured", {})

        dev = clinical_structured.get("development_factors", {})
        peri = clinical_structured.get("perinatal_risk", {})
        phys = clinical_structured.get("physiological_status", {})

        def format_section(d):
            if not d:
                return "  (not provided)"
            lines = []
            for k, v in d.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        # token级的
        developmental_block = format_section(dev)
        perinatal_block = format_section(peri)
        physiological_block = format_section(phys)


        prompt = self.prompt_clinic

        required_placeholders = [
            "__IMAGE_TOKENS__",
            "__DEV_FACTORS__",
            "__PERINATAL_RISK__",
            "__PHYS_STATUS__",
        ]

        for ph in required_placeholders:
            if ph not in prompt:
                raise ValueError(f"Prompt template missing {ph} placeholder")

        prompt = (
            prompt
            .replace("__IMAGE_TOKENS__", image_tokens)
            .replace("__DEV_FACTORS__", developmental_block)
            # .replace("__DEV_FACTORS__",   "(not provided)")
            # .replace("__PERINATAL_RISK__", "(not provided)")
            # .replace("__PHYS_STATUS__", "(not provided)")
            .replace("__PERINATAL_RISK__", perinatal_block)
            .replace("__PHYS_STATUS__", physiological_block)
        )
        return prompt

    def load_data(self):
        dataset_path = self.dataset_path
        self.full_samples = [] 
        json_path = os.path.join(dataset_path,"test.json")

        with open(json_path,"r") as f:
            dataset = json.load(f)
        
        skipped_count = 0 # 用于统计

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                
                # image_root = self.dataset_path
                image_root="/data01/kjt_data/MLLM/data/NeoCXR/images"

                image_field = sample.get("image") 
                self.full_samples.append(sample.copy())
                relative_images = [p.strip() for p in str(image_field).split(",") if p.strip()]
                

                full_image_paths = [os.path.join(image_root, p) for p in relative_images if p]
                
                if not all(os.path.exists(p) for p in full_image_paths):
                    skipped_count += 1
                    continue 

                try:
                    images_pil = [Image.open(p).convert('RGB') for p in full_image_paths]
                except Exception as e:

                    print(f"WARNING: Failed to open image at {full_image_paths[0]} due to: {e}. Skipping.")
                    skipped_count += 1
                    continue

                sample['images_pil'] = images_pil
                
                sample = self.construct_messages(sample)
                
                self.samples.append(sample)



    def construct_messages(self, sample):

        images = sample['images_pil'] 
        
        # prompt = sample["conversation_en"][0]["value"]
        prompt=self.build_prompt(sample)

        messages = {"prompt":prompt,"images":images}
        sample["messages"] = messages
        
        # 移除 'images_pil' 键以保持样本干净 (可选)
        del sample['images_pil'] 
        
        return sample


    def cal_metrics(self,out_samples):
        import pandas as pd

        predictions_data = []
        ground_truth_data = []
        

        # for i,sample in enumerate(out_samples):
        #     response = sample["response"]
        #     golden = f"Imaging conclusion: {sample['imaging_conclusion_en']} Disease diagnosis: {sample['disease_diagnosis_en']}"

        #     study_id = f"study_{i+1}"
            
        #     predictions_data.append({
        #         'study_id': study_id,
        #         'report': response
        #     })

        #     ground_truth_data.append({
        #         'study_id': study_id,
        #         'report': golden
        #     })

        for i, (out_sample, full_sample) in enumerate(zip(out_samples, self.full_samples)):
            response = out_sample["response"]
            # 从 full_sample 中提取 golden
            imaging_conclusion = full_sample.get("imaging_conclusion_en", "").strip()
            disease_diagnosis = full_sample.get("disease_diagnosis_en", "").strip()
            golden = f"Imaging conclusion: {imaging_conclusion} Disease diagnosis: {disease_diagnosis}"

            study_id = f"study_{i+1}"

            predictions_data.append({'study_id': study_id, 'report': response})
            ground_truth_data.append({'study_id': study_id, 'report': golden})

        total_samples_loaded = len(out_samples)

        predictions_df = pd.DataFrame(predictions_data)
        ground_truth_df = pd.DataFrame(ground_truth_data)

        prediction_path = os.path.join(self.output_path,'predictions.csv')
        ground_truth_path = os.path.join(self.output_path,'ground_truth.csv')

        predictions_df.to_csv(prediction_path, index=False)
        ground_truth_df.to_csv(ground_truth_path, index=False)

        # return {"total metrics":"please use cal_report_metrics.py to generate metrics"},out_samples
        return {
            "total metrics": "please use cal_report_metrics.py to generate metrics",
            "total_valid_samples": total_samples_loaded 
        }, out_samples

                