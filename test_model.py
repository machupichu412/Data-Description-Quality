#!/usr/bin/env python3

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# --- Config ---
model_path = "phi_finetune_output_gen"  # Directory where your fine-tuned model and adapter are saved
test_file = "test_ALL_no_labels_v3.jsonl"  # Your JSONL test file with the "input" field
output_file = "inference_results_labeled.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# Add special <|end|> token if not present
tokenizer.add_special_tokens({"eos_token": "<|end|>"})
eos_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
if eos_token_id is None or eos_token_id == tokenizer.unk_token_id:
    raise ValueError("❌ '<|end|>' token not found in tokenizer.")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- Load base model and update embeddings ---
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
# Resize embeddings to match the updated tokenizer vocabulary
base_model.resize_token_embeddings(len(tokenizer))

# --- Load the adapter-wrapped model (avoid merging adapter weights) ---
model = PeftModel.from_pretrained(base_model, model_path)
# Resize again after wrapping adapter if needed
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval()

# --- Load dataset ---
# The test file is expected to have an "input" field with your prompt.
dataset = load_dataset("json", data_files=test_file, split="train")
prompts = dataset["input"]

# --- Inference loop ---
results = []

for i, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # Adjust as needed for full responses
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=False,
            num_beams=1,
        )

    # Only decode the generated tokens (beyond the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Simple binary classification based on response text
    if "pass" in response.lower():
        label = 1
    elif "fail" in response.lower():
        label = 0
    else:
        label = -1  # For unclear outputs

    print(f"[{i+1}] {response}")
    results.append({
        "index": i + 1,
        "prompt": prompt,
        "output": response,
        "label": label
    })

# --- Save results ---
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Inference results saved to: {output_file}")