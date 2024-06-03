#!/usr/bin/env python
# coding: utf-8


import json
import logging
import os
from pathlib import Path


import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import logging as transformers_logging

torch.random.manual_seed(0)

transformers_logging.set_verbosity_error()

ROOT = Path.cwd()
print(ROOT)



# get environment variables
model_id = os.environ.get("REPO_ID")
model_name = model_id.replace("/", "_")
prompt_strategy = os.environ.get("PROMPT_STRATEGY")
idk = os.environ.get("IDK", 'True')
idk = True if idk == 'True' else False
input_path = os.environ.get("INPUT_PATH", ROOT / "data" / "legal_hallucination_prompts_n_1_10.jsonl")
icl_num = os.environ.get("ICL_NUM", 5)

# Set up logging to file 
log_dir = ROOT / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / f"{prompt_strategy}_{"idk" if idk else "ik"}_n_{icl_num}_{model_name}.log"),
        logging.StreamHandler(),
    ],
)

logging.info(f"Model ID: {model_id}")
logging.info(f"Prompt Strategy: {prompt_strategy}")
if prompt_strategy == 'fs':
    logging.info(f"ICL Number: {icl_num}")
logging.info(f"IDK: {idk}")
logging.info(f"Input Path: {input_path}")


items = pd.read_json(input_path, lines=True)
logging.info(f"Loaded {len(items)} items")






if idk:
    system_prompt = """Your task is to answer legal questions. Provide a concise answer without an explanation. If you don't know the answer, respond with "I don't know"."""
else:
    system_prompt = """Your task is to answer legal questions. Provide a concise answer without an explanation."""
logging.info(f"System prompt: {system_prompt}")


tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token_id=tokenizer.eos_token_id
logging.info(f"Loaded tokenizer {model_id}")


# set output path 

output_path = ROOT / "outputs" / f"{prompt_strategy}_{"idk" if idk else "ik"}_n_{icl_num}_{model_name}.jsonl"
logging.info(f"Output path: {output_path}")
# create directory if it does not exist
output_path.parent.mkdir(exist_ok=True, parents=True)

# load outputs if file exists
if output_path.exists():
    outputs = pd.read_json(output_path, lines=True)
    logging.info(f"Loaded {len(outputs)} outputs")

    # filter items that have not been processed basesd on the question value
    items = items[~items['question'].isin(outputs['question'])]
    logging.info(f"Processing {len(items)} items")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto",
    trust_remote_code=True,
    revision="main",
    quantization_config=bnb_config
    )
logging.info(f"Loaded model {model_id}")

fs_prompt_col = f'few_shot_{icl_num}' 

if prompt_strategy == 'zs':
    prompt_template = """Question: {question}"""
    template_func = lambda row: prompt_template.format(question=row['question'])
elif prompt_strategy == 'zs_cot':
    prompt_template = """Question: {question} Let's think step by step."""
    template_func = lambda row: prompt_template.format(question=row['question'])
elif prompt_strategy == 'fs':
    prompt_template = """Examples:
    ```
    {examples}
    ```
    Question: {question}
    """
    template_func = lambda row: prompt_template.format(examples=row[fs_prompt_col], question=row['question'])
elif prompt_strategy == 'fs_cot':
    prompt_template = """Examples:
    ```
    {examples}
    ```
    Question: {question} Let's think step by step."""
    template_func = lambda row: prompt_template.format(examples=row[fs_prompt_col], question=row['question'])

items['prompt'] = items.apply(
    template_func,
    axis=1
)

def create_messages(item, model_name=model_name):
    if 'Mixtral' in model_name or 'Mistral' in model_name:
        item_content = f"{system_prompt}\n{item['prompt']}"
        messages = [
            {"role": "user", "content": item_content},
            {"role": "assistant", "content": '\nAnswer: '},
        ]
    elif 'Phi' in model_name:
        item_content = f"{system_prompt}\n{item['prompt']}"
        messages = [
            {"role": "user", "content": item_content}
        ]
    elif 'Llama' in model_name:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item['prompt']},
        ]
    elif 'gemma' in model_name or 'saul' in model_name.lower():
        item_content = f"{system_prompt}\n{item['prompt']}"
        messages = [
            {"role": "user", "content": item_content},
            # {"role": "model", "content": '\nAnswer: '},
        ]
    else:
        item_content = f"{system_prompt}\n{item['prompt']}"
        messages = [
            {"role": "user", "content": item_content},
            {"role": "assistant", "content": '\nAnswer: '},
        ]
    return messages


items = items.sample(frac=1, random_state=42)


def generate_object(item, prompt_strategy=prompt_strategy):
    if 'cot' in prompt_strategy:
        max_new_tokens = 100
    else:
        max_new_tokens = 50
    messages = create_messages(item)
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text


for i, item in tqdm(items.iterrows(), total=len(items)):
    output = {}
    output['qid'] = item['qid']
    output['question'] = item['question']
    output['predicate'] = item['predicate']
    output_text = generate_object(item)
    output['output_text'] = output_text
    with open(output_path, 'a') as f:
        f.write(json.dumps(output) + '\n')