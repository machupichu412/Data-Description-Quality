#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json

# --- Config ---
model_path = "phi_finetune_output_gen"
test_file = "test_ALL_v2.jsonl"
output_file = "inference_results.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Ensure EOS token exists
tokenizer.add_special_tokens({"eos_token": "<|end|>"})
eos_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
if eos_token_id in [None, tokenizer.unk_token_id]:
    raise ValueError("❌ '<|end|>' token not found in tokenizer.")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
model.resize_token_embeddings(len(tokenizer))  # ensure new tokens are used
model = model.to(device)
model.eval()

# --- Load dataset ---
dataset = load_dataset("json", data_files=test_file, split="train")

# --- Prompt builder ---
def extract_prompt(example):
    return f"<|system|>{example['system_prompt']}<|user|>{example['user_input']}<|assistant|>"

dataset = dataset.map(lambda x: {"prompt": extract_prompt(x)})
prompts = dataset["prompt"]

# --- Inference ---
results = []

for i, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=True,                   # sampling encourages diversity
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"[{i+1}] {response}\n")
    results.append({
        "index": i + 1,
        "output": response,
        "prompt": prompt
    })

# --- Save to JSON ---
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Inference results saved to: {output_file}")
