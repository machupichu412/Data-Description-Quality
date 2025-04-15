#!/usr/bin/env python3

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Config ---
base_model_path = "/home/cicconel/phi4_mini_instruct_local"
dataset_path = "/home/cicconel/llama_models/train_ALL_v3.jsonl"
output_dir = "phi_finetune_output_gen"

# --- LoRA config ---
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj", "o_proj"]
)

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.add_special_tokens({"eos_token": "<|end|>"})
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))

# --- Prepare for LoRA ---
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# --- Load and preprocess dataset ---
dataset = load_dataset("json", data_files=dataset_path, split="train")

def is_valid(example):
    return isinstance(example.get("input"), str) and isinstance(example.get("output"), str)

dataset = dataset.filter(is_valid)

def format_example(example):
    full_text = example["input"] + example["output"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# --- Training setup ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=True,
    learning_rate=2e-5,
    max_grad_norm=0.3,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
    deepspeed="deepspeed_config.json",
    optim="paged_adamw_32bit",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("🚀 Starting training...")
trainer.train()

# --- Save LoRA adapter and tokenizer using the Trainer ---
print("💾 Saving model and tokenizer...")

# Instead of calling model.save_pretrained(), use the trainer's save_model() method.
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Done. Model and tokenizer saved to: {output_dir}")