#!/usr/bin/env python3
# Fine-tunes a LLaMA model with LoRA to generate "Pass/Fail + Reason" outputs

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# --- Config ---
base_model_path = "/home/cicconel/llama3_8B_local"           # Path to base LLaMA model
dataset_path = "/home/cicconel/llama_models/train.jsonl"     # Training data
output_dir = "llama_finetune_output_gen"                     # Save directory

# LoRA config
lora_config = LoraConfig(
    r=1,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# --- Load tokenizer and model ---
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Format dataset into prompt/response pairs ---
def format_example(example):
    user_msg, assistant_msg = "", ""
    for msg in example["messages"]:
        if msg["role"] == "user":
            user_msg = msg["content"].strip()
        elif msg["role"] == "assistant":
            assistant_msg = msg["content"].strip()
    prompt = f"<|user|> {user_msg}\n<|assistant|> {assistant_msg}"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Loading and formatting dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# --- Data collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Training args ---
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
    optim="paged_adamw_32bit"
)

# --- Train ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

print("Saving model and tokenizer...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)