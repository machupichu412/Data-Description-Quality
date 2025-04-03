#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import json

# Paths
base_model_path = "/home/cicconel/phi4_mini_instruct_local"
peft_model_path = "phi_finetune_output_gen"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32, trust_remote_code=True)
model = PeftModel.from_pretrained(model, peft_model_path)
model = model.to(device)
model.eval()

# Load dataset
dataset = load_dataset("json", data_files="train_ALL.jsonl", split="train")

# Build Phi-4-style prompt with system + user + assistant
def extract_chat_prompt(example):
    return example["text"]


# Apply to dataset
dataset = dataset.map(lambda x: {"prompt": extract_chat_prompt(x)})
prompts = dataset["prompt"]

# Inference loop
results = []

for i, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated.split("<|assistant|>")[-1].strip()

    print(f"[{i+1}] {response}\n")
    results.append({"index": i+1, "output": response, "prompt": prompt})

# Save outputs
with open("inference_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Inference results saved to inference_results.json")

