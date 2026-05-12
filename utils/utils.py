import json
import math
import sys
import copy
import re
import os
import json
import difflib
import asyncio
import random
# import sys
# sys.setrecursionlimit(10000)

#这里想看一下多少样本没有成功调用api
# import json
# from datetime import datetime

# # 在文件开头添加（如果还没有）
# FAILED_RECORDS = []  # 全局变量记录失败记录

# def save_failed_records(filename='failed_api_calls.json'):
#     """将失败记录保存到 JSON 文件"""
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(FAILED_RECORDS, f, indent=2, ensure_ascii=False)
#     print(f"✅ 已保存 {len(FAILED_RECORDS)} 条失败记录到 {filename}")

# def generate_outputs(self, messages, sample_ids=None, **kwargs):
#     """
#     批量生成输出，并记录失败样本
    
#     Args:
#         messages: 消息列表
#         sample_ids: 可选，每个样本的唯一ID列表（如 ["55504914", "55504915", ...]）
#         **kwargs: 其他参数
    
#     Returns:
#         results: 成功的结果列表
#         failed_indices: 失败的索引列表
#     """
#     failed_indices = []  # 记录失败的索引
#     results = []
    
#     async def deal_tasks(tasks):
#         return await asyncio.gather(*tasks)

#     async def generate_output_async_with_tracking(idx, message, sample_id=None, **kwargs):
#         try:
#             response = await self.response_async(message, **kwargs)
#             return idx, response, None, sample_id  # 成功：返回响应、None错误、sample_id
#         except Exception as e:
#             error_msg = str(e)
#             print(f"❌ API 调用失败 - 样本 ID: {sample_id}, 索引: {idx}, 错误: {error_msg}")
            
#             # 记录到全局失败列表
#             FAILED_RECORDS.append({
#                 "id": sample_id or f"unknown_{idx}",
#                 "index": idx,
#                 "timestamp": datetime.now().isoformat(),
#                 "error": error_msg,
#                 "message": str(message)  # 可选：记录原始消息内容
#             })
            
#             return idx, None, error_msg, sample_id  # 失败：返回None和错误信息

#     # 如果没有提供 sample_ids，创建默认ID
#     if sample_ids is None:
#         sample_ids = [f"sample_{i}" for i in range(len(messages))]
    
#     # 创建任务
#     tasks = [
#         generate_output_async_with_tracking(i, messages[i], sample_ids[i], **kwargs) 
#         for i in range(len(messages))
#     ]
    
#     raw_results = asyncio.run(deal_tasks(tasks))
    
#     # 处理结果
#     for idx, response, error, sample_id in raw_results:
#         if error is None:
#             results.append(response)
#         else:
#             results.append(None)  # 或者添加默认值
#             failed_indices.append(idx)
    
#     # 输出失败信息
#     if failed_indices:
#         print(f"⚠️  共有 {len(failed_indices)} 个样本调用API失败")
#         print(f"📌 失败样本索引: {failed_indices}")
#         print(f"📌 失败样本ID: {[sample_ids[i] for i in failed_indices]}")
        
#         # 保存失败记录
#         save_failed_records()
    
#     return results, failed_indices


# from google import genai
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from mathruler.grader import extract_boxed_content

from collections import defaultdict, Counter
from openai import AzureOpenAI, OpenAI,AsyncAzureOpenAI,AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)

def tokenize(text):
    text = text.lower().replace(".", " .").split(" ")
    return text

def bleu(pred,target,n):
    weights=[1/n for _ in range(n)]
    tokenized_target = tokenize(target)
    tokenized_pred = tokenize(pred)
    return sentence_bleu([tokenized_target], tokenized_pred, weights=weights)

def rouge(pred,target):
    rouge_scorer = Rouge()
    rouge_scores = rouge_scorer.get_scores(pred.lower(), target.lower())
    return rouge_scores


def get_compare_messages(question,response,answer):
    prompt = f"""
Your task is to determine whether the user's answer is correct based on the provided questions and standard answers (for example, if the user expresses a similar meaning to the standard answer, or another interpretation of the standard answer, it is considered correct.)

The question is: {question}

The standard answer: {answer}

The user's answer: {response}

Please strictly follow the following format for output(0 represents correct, 1 represents incorrect):
<think>{{your concise think step}}</think>
<judge>{{0/1}}</judge>

for example:
<think>The standard answer is right, and the user's answer is right frontal lobe, they express the same meaning, so it is correct.</think>
<judge>0</judge>
    """
    messages = [{"role":"user","content":prompt}]
    return messages


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = 0
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index

def judge_multi_choice(choices,answer,response,alphas = None):
    response = response.lower()
    if response.split("\n\n")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[0]
    elif response.split("\n\n")[-1].split(".")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[-1].split(".")[0]
    
    response = parse_response(response)
    alphas = [chr(ord('a') + i) for i in range(len(choices))]
    choices = [choice.lower() for choice in choices]
    flag = False
    response = response.strip().lower()
    response = response.replace("\n","")
    split_response = response.split(".")[0]
    split_response = split_response.split(":")[-1]
    answer = answer.strip().lower()
    
    if len(split_response) > 300:
        flag = False
    # letter,letter.  choice,choice
    if split_response == answer:
        flag = True
    
    # letter,choice
    elif split_response in alphas:
        if choices[ord(split_response)-ord("a")]== answer:
            flag = True
    
    # choice letter
    elif split_response in choices:
        if answer in alphas and split_response == choices[ord(answer)-ord("a")]:
            flag = True
    # unparsed
    else:
        index = find_most_similar_index(choices,response)
        if alphas[index] == answer or choices[index] == answer:
            flag = True
    return flag


def parse_response(response):
    response = response.lower()
    ### lynadd
    response = response.replace("<s>", "").replace("</s>", "").strip()
    # # 移除USER和图像标记
    # if "user:" in response:
    #     response = response.split("user:")[0].strip()
    # response = response.replace("<image>", "").strip()
    ### lynadd
    if "boxed" in response:
        response = extract_boxed_content(response)
    elif "<answer>" in response:
        response = extract(response,"answer")
    # 提取ASSISTANT后的内容
    elif "assistant:" in response:
        response = response.split("assistant:")[-1].strip()
    answer_patterns = [
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is",
        "answer:",
        "答案:",
        "final answer",
        "final answer is"
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in response:
            response = response.split(answer_pattern)[-1]
    
    return response

def judge_close_end_vqa(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    response = response.replace("\n","").replace(".","")
    if response == answer:
        return True
    else:
        return False

def judge_judgement(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    response = response.replace("\n","").replace(".","")
    if ('yes' in response) ^ ('no' in response):
        if answer in response:
            return True
    return False


def judge_open_end_vqa(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    bleu1 = bleu(response,answer,1)
    bleu2 = bleu(response,answer,2)
    bleu3 = bleu(response,answer,3)
    bleu4 = bleu(response,answer,4)

    em = response == answer
    rouge_scores = rouge(response,answer)
    rouge_1 = rouge_scores[0]["rouge-1"]["f"]
    rouge_2 = rouge_scores[0]["rouge-2"]["f"]
    rouge_l = rouge_scores[0]["rouge-l"]["f"]

    precision,recall,f1 = calculate_f1(response,answer)


    return {
        "em" : em,
        "bleu1" : bleu1,
        "bleu2" : bleu2,
        "bleu3" : bleu3,
        "bleu4" : bleu4,
        "rouge1" : rouge_1,
        "rouge2" : rouge_2,
        "rougel" :  rouge_l,
        "precision": precision,
        "recall": recall,
        "f1" :f1         
    }


def calculate_f1(prediction, ground_truth):
    prediction_tokens = set(prediction.lower().split())
    ground_truth_tokens = set(ground_truth.lower().split())
    
    common = prediction_tokens & ground_truth_tokens
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0
    
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        return 0,0,0
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1,precision,recall


def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type,hard = True):
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            return target_str
        elif hard:
            return text
        else:
            return ""
    else:
        return ""

# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ds, f, indent=4, ensure_ascii=False)

class fake_response:
    def __init__(self,usage):
        self.usage = usage

def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

async def deal_tasks(tasks, max_concurrent_tasks=500):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results = []

    async def sem_task(task):
        async with semaphore:
            return await task  

    sem_tasks = [sem_task(task) for task in tasks]

    for coro in tqdm_asyncio.as_completed(sem_tasks, total=len(sem_tasks)):
        result = await coro
        results.append(result)

    return results

import requests

# def before_retry_fn(retry_state):
#     print(f"Retrying... attempt {retry_state.attempt_number}")
class deepseek:
    def __init__(self, model=None):
        if model is None:
            model = os.environ.get("judge_model", "DeepSeek-R1")
            print("model",model)

        base_url = os.environ.get("base_url", "https://llmapi.tongji.edu.cn/v1")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get("api_key")
        # Set api_key through the `api_key` environment variable.

        if not self.api_key:
            raise ValueError("No API key found. Please set environment variable `api_key`")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3), before=before_retry_fn)
    def response(self, messages, **kwargs):
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "n": kwargs.get("n", 1),
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 4000),
        }
        # print("modelka",kwargs.get("model", self.model))
        url = f"{self.base_url}/chat/completions"
        # # url="https://openrouter.ai/api/v1/chat/completions"
        # print("url",url)
        # sys.exit() 
        r = requests.post(url, headers=self.headers, json=payload, timeout=kwargs.get("timeout", 180))
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3), before=before_retry_fn)
    async def response_async(self, messages, **kwargs):
        # 用 httpx 进行异步请求
        import httpx
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "n": kwargs.get("n", 1),
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 4000),
        }
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=kwargs.get("timeout", 180)) as client:
            r = await client.post(url, headers=self.headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

    def generate_output(self, messages, **kwargs):
        try:
            response = self.response(messages, **kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
            import traceback
            traceback.print_exc()
        if hasattr(e, 'response'):
            print("Status:", e.response.status_code)
            print("Body:", e.response.text)
        return response

    async def generate_output_async(self, idx, messages, **kwargs):
        try:
            response = await self.response_async(messages, **kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return idx, response

    def generate_outputs(self, messages, **kwargs):
        async def deal_tasks(tasks):
            return await asyncio.gather(*tasks)

        tasks = [self.generate_output_async(i, messages[i], **kwargs) for i in range(len(messages))]
        results = asyncio.run(deal_tasks(tasks))
        results = sorted(results, key=lambda x: x[0])
        results = [x[1] for x in results]
        return results


class openrouter:
    def __init__(self, model=None):
        if model is None:
            model = os.environ.get("judge_model", "openai/gpt-4.1")
            print("model",model)

        base_url = os.environ.get("base_url", "https://openrouter.ai/api/v1")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get("api_key")
        # Set api_key through the `api_key` environment variable.

        if not self.api_key:
            raise ValueError("No API key found. Please set environment variable `api_key`")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3), before=before_retry_fn)
    def response(self, messages, **kwargs):
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "n": kwargs.get("n", 1),
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 4000),
        }
        # print("modelka",kwargs.get("model", self.model))
        url = f"{self.base_url}/chat/completions"
        # # url="https://openrouter.ai/api/v1/chat/completions"
        # print("url",url)
        # sys.exit() 
        r = requests.post(url, headers=self.headers, json=payload, timeout=kwargs.get("timeout", 180))
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3), before=before_retry_fn)
    async def response_async(self, messages, **kwargs):
        # 用 httpx 进行异步请求
        import httpx
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "n": kwargs.get("n", 1),
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 4000),
        }
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=kwargs.get("timeout", 180)) as client:
            r = await client.post(url, headers=self.headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

    def generate_output(self, messages, **kwargs):
        try:
            response = self.response(messages, **kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
            import traceback
            traceback.print_exc()
        if hasattr(e, 'response'):
            print("Status:", e.response.status_code)
            print("Body:", e.response.text)
        return response

    async def generate_output_async(self, idx, messages, **kwargs):
        try:
            response = await self.response_async(messages, **kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return idx, response

    def generate_outputs(self, messages, **kwargs):
        async def deal_tasks(tasks):
            return await asyncio.gather(*tasks)

        tasks = [self.generate_output_async(i, messages[i], **kwargs) for i in range(len(messages))]
        results = asyncio.run(deal_tasks(tasks))
        results = sorted(results, key=lambda x: x[0])
        results = [x[1] for x in results]
        return results

class openai_llm:
    def __init__(self,model = None):
        if model is None:
            model = os.environ.get("judge_model","gpt-4.1-2025-04-14")
        
        base_url = os.environ.get("base_url", "https://api.openai.com/v1")
        base_url = None if base_url == "None" else base_url

        self.model = model

        api_key = os.environ["api_key"]

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
            )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
            )
    
    @retry(wait=wait_fixed(1000), stop=stop_after_attempt(1000), before=before_retry_fn)
    def response(self,messages,**kwargs):
        response = self.client.chat.completions.create(
            # gpt4o-0513  gpt4-turbo-2024-04-29 gpt-4o-2
            model=kwargs.get("model", self.model),
            messages=messages,
            n = kwargs.get("n", 1),
            temperature= kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4000),
            timeout=kwargs.get("timeout", 180)
        )
        return response.choices[0].message.content
    
    @retry(wait=wait_fixed(1000), stop=stop_after_attempt(1000), before=before_retry_fn)
    async def response_async(self,messages,**kwargs):
        response = await self.async_client.chat.completions.create(
            # gpt4o-0513  gpt4-turbo-2024-04-29 gpt-4o-2
            model=kwargs.get("model", self.model),
            messages=messages,
            n = kwargs.get("n", 1),
            temperature= kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4096),
            timeout=kwargs.get("timeout", 180)
        )      
        return response.choices[0].message.content
    
    def generate_output(self,messages,**kwargs):
        try:
            response = self.response(messages,**kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return response
    
    async def generate_output_async(self,idx, messages,**kwargs):
        try:
            response = await self.response_async(messages,**kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return idx,response
    
    def generate_outputs(self,messages,**kwargs):
        tasks = [self.generate_output_async(i,messages[i],**kwargs) for i in range(len(messages))]
        results = asyncio.run(deal_tasks(tasks))
        results = sorted(results, key=lambda x: x[0])
        results = [x[1] for x in results]
        return results

class Gemini:
    def __init__(self,model = "gemini-2.0-flash") -> None:
        self.model = model
        self.api_key = os.environ.get("api_key")

        self.client = genai.Client(api_key=self.api_key)
    
    @retry(wait=wait_fixed(1000), stop=stop_after_attempt(3000), before=before_retry_fn)
    def response(self,messages,**kwargs):
        response = self.client.models.generate_content(
            model=kwargs.get("model", self.model),
            contents=messages,
        )
        return response.text
    
    def generate_output(self,messages,**kwargs):
        try:
            response = self.response(messages,**kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return response
    
    def generate_outputs(self,messages,**kwargs):
        results = []
        for message in tqdm(messages):
            response = self.generate_output(message,**kwargs)
            results.append(response)
        return results



def init_judger():
    if os.environ.get("judge_model_type", "openai") in ["openai","claude","deepseek"]:
        judger = openai_llm()
    elif os.environ.get("judge_model_type", "openai") == "gemini":
        judger = Gemini()
    elif os.environ.get("judge_model_type", "openai") == "openrouter":
        judger = openrouter()
    elif os.environ.get("judge_model_type", "openai") == "deepseek":
        judger = deepseek()
    else:
        raise ValueError("Unsupported judge model type. Please set 'judge_model_type' to 'openai', 'gemini', or 'claude'.")
    return judger

judger = init_judger()
