import torch
import os
from PIL import Image
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoTokenizer,AutoProcessor, LlavaForConditionalGeneration
# from llava.constants import DEFAULT_IMAGE_TOKEN  # 来自官方 LLaVA 仓库
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
class Llava:
    def __init__(self,model_path,args):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)

        self.llm = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,device_map="auto", low_cpu_mem_usage=True)
        # self.llm.to("cuda")

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def process_messages(self, messages):

        if isinstance(messages, list):
            prompts = []
            images_batch = []

            for msg in messages:
                parts = []
                if msg.get("system"):
                    parts.append(str(msg["system"]).strip())
                if msg.get("prompt"):
                    parts.append(str(msg["prompt"]).strip())
                prompt = "\n".join(parts) if parts else ""
                prompts.append(prompt)

                if "image" in msg:
                    raw = msg["image"]
                    if isinstance(raw, str):
                        img = Image.open(raw).convert("RGB")
                    else:
                        img = raw.convert("RGB")
                    images_batch.append(img)
                elif "images" in msg:
                    tmp_imgs = []
                    for raw in msg["images"]:
                        if isinstance(raw, str):
                            img = Image.open(raw).convert("RGB")
                        else:
                            img = raw.convert("RGB")
                        tmp_imgs.append(img)
                    images_batch.append(tmp_imgs)
                else:
                    images_batch.append(None)

            llm_inputs = self.processor(
                text=prompts,
                images=images_batch,
                padding=True,
                return_tensors="pt",
            ).to(self.llm.device)

            return llm_inputs  

        parts = []
        has_image = False
        if messages.get("system"):
            parts.append(str(messages["system"]).strip())
        if messages.get("prompt"):
            parts.append(str(messages["prompt"]).strip())
        prompt = "\n".join(parts) if parts else ""

        images_for_processor = None

        if "image" in messages:
            raw = messages["image"]
            img = Image.open(raw).convert("RGB") if isinstance(raw, str) else raw.convert("RGB")
            images_for_processor = img
            has_image = True

        elif "images" in messages:
            imgs = []
            for raw in messages["images"]:
                img = Image.open(raw).convert("RGB") if isinstance(raw, str) else raw.convert("RGB")
                imgs.append(img)
            images_for_processor = imgs
            has_image = True
        
        # if has_image:
        #     for _ in images_for_processor:
        #         prompt += "<image>\n"


        llm_inputs = self.processor(
            text=prompt,
            images=images_for_processor,
            padding=True,
            return_tensors="pt",
        ).to(self.llm.device)

        return llm_inputs

    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        do_sample = False if self.temperature == 0 else True
        outputs = self.llm.generate(**llm_inputs,max_new_tokens=self.max_new_tokens, do_sample=do_sample,repetition_penalty=self.repetition_penalty,temperature=self.temperature,top_p = self.top_p)
        outputs = self.processor.decode(outputs[0])
        return outputs
    
    def generate_outputs(self,messages_list):
        res = []
        for messages in messages_list:
            result = self.generate_output(messages)
            res.append(result)
        return res
