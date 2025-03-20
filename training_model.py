#!/usr/bin/env python3
# Optimized LLaMA fine-tuning script for distributed GPU training using DeepSpeed
# Uses LoRA (Low-Rank Adaptation) with minimal rank and DeepSpeed ZeRO optimization to reduce memory usage.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuration – adjust paths and names as needed
model_name = "/home/cicconel/llama3_8B_local"  # Path to your pre-trained LLaMA model (8B parameters)
train_dataset_name = "/home/cicconel/llama_models/train.jsonl"  # Local JSONL file containing your training data
output_dir = "llama_finetune_output"             # Directory to save model checkpoints

# LoRA configuration: low rank to minimize memory footprint
lora_rank = 1
lora_alpha = 16
lora_dropout = 0.05

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model WITHOUT device_map so that DeepSpeed can handle distributed training.
# Load model normally
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2, 
    torch_dtype=torch.float16,  
    device_map="auto",  # Ensure automatic mapping
    trust_remote_code=True,    
)

# Ensure all model parameters are moved to CUDA explicitly
model = model.to(device)

# Print device details for debugging
for name, param in model.named_parameters():
    if param.device != torch.device(device):
        print(f"⚠️ Parameter {name} is on {param.device}, moving to {device}.")
        param.to(device)

model.gradient_checkpointing_enable()

# Apply LoRA to the model so that only a small fraction of parameters are trainable.
peft_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Note: Do not call model.to("cuda") here; DeepSpeed will manage device placement.

# Load and preprocess the dataset.
print("Loading and tokenizing dataset...")
# Load JSONL dataset; each example is expected to have a "messages" field.
dataset = load_dataset("json", data_files=train_dataset_name, split="train")

def tokenize_function(examples):
    texts = []
    for msg_list in examples["messages"]:
        # Extract the second message's content (assuming that is the user message)
        if isinstance(msg_list, list) and len(msg_list) > 1:
            texts.append(msg_list[1]["content"])
        else:
            texts.append("")
    return tokenizer(texts, return_attention_mask=True, truncation=True, max_length=512)

dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Data collator: pads sequences and creates labels equal to input_ids (for causal LM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments with DeepSpeed integration for efficient memory usage
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,                        # Adjust epochs as needed
    per_device_train_batch_size=1,             # Minimal batch size to reduce memory footprint
    gradient_accumulation_steps=8,             # Effective batch size = 1 * 8 = 8
    gradient_checkpointing=True,               # Saves memory by recomputing activations
    fp16=True,                                 # Mixed precision training reduces memory usage
    learning_rate=2e-5,                        # Adjust learning rate as needed
    max_grad_norm=0.3,                         # Gradient clipping for stability
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
    deepspeed="deepspeed_config.json",         # Path to your DeepSpeed config file
)

# Clear any cached GPU memory before starting training
torch.cuda.empty_cache()

# Initialize Trainer with our model, dataset, and training configuration.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

print("Saving fine-tuned model...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

