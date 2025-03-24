#!/usr/bin/env python3
# Optimized LLaMA fine-tuning script for distributed GPU training using DeepSpeed
# Uses LoRA (Low-Rank Adaptation) with minimal rank and DeepSpeed ZeRO optimization to reduce memory usage.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DefaultDataCollator
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()

# Apply LoRA
peft_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="SEQ_CLS"
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
    labels = []

    for msg_list in examples["messages"]:
        # Extract user input
        user_msg = ""
        assistant_msg = ""

        if isinstance(msg_list, list):
            for msg in msg_list:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_msg = msg["content"]

        # Prepare input text
        texts.append(user_msg)

        # Extract label from assistant message
        if assistant_msg.lower().startswith("pass"):
            labels.append(1)
        elif assistant_msg.lower().startswith("fail"):
            labels.append(0)
        else:
            labels.append(-1)  # Optional: for debugging bad data

    # Tokenize user message only (sequence classification task)
    tokenized = tokenizer(texts, truncation=True, padding=True, max_length=512)
    tokenized["labels"] = labels
    return tokenized


dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Data collator: pads sequences and creates labels equal to input_ids (for causal LM)
data_collator = DefaultDataCollator()

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
    deepspeed="deepspeed_config.json",
    optim="paged_adamw_32bit"
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