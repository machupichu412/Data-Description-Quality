#!/usr/bin/env python3
# Fine-tunes Phi-4-mini-instruct using LoRA to generate "Reason + Decision" outputs

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from peft.utils.other import prepare_model_for_kbit_training

# --- Config ---
base_model_path = "/home/cicconel/phi4_mini_instruct_local"
dataset_path = "/home/cicconel/llama_models/train_ALL_v2.jsonl"  # Updated dataset path
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
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

for name, module in model.named_modules():
    print(name)

model = prepare_model_for_kbit_training(model)  # FIRST
model = get_peft_model(model, lora_config)      # THEN apply LoRA
model.enable_input_require_grads()              # Enable gradient tracking
model.print_trainable_parameters()

# --- Load dataset ---
print("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- Validate and filter dataset ---
def is_valid(example):
    return (
        isinstance(example.get("system_prompt"), str) and
        isinstance(example.get("user_input"), str) and
        isinstance(example.get("reason"), str) and
        isinstance(example.get("decision"), str)
    )

dataset = dataset.filter(is_valid)
print(f"✅ Loaded {len(dataset)} valid examples")

# --- Format dataset into Phi-4 chat-style prompt/response ---
def format_example(example):
    prompt = f"<|system|>{example['system_prompt']}<|user|>{example['user_input']}<|assistant|>"
    response = f"Reason: {example['reason']} Decision: {example['decision']}<|end|>"
    full_text = prompt + response

    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


print("Tokenizing dataset...")
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


