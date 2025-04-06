import json
from accelerate import Accelerator
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch

def process_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)

            input_text = entry["system_prompt"] + " " + entry["user_input"]
            target_text = f"Reason: {entry['reason']} Decision: {entry['decision']}"

            # Model expects concatenated input plus output
            full_text = input_text + " " + target_text

            data.append({
                "full_text": full_text
            })

    return data

accelerate = Accelerator();

# Load data
print("Loading the data")
file_path = 'data_new/train_ALL_v2.jsonl'  # Path to your JSONL file
processed_data = process_jsonl(file_path)

# Convert to dataset
dataset = Dataset.from_list(processed_data)

local_model_path = "phi-4-local"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

def tokenize_function(examples):
    return tokenizer(
        examples["full_text"],
        padding="max_length",
        truncation=True,
        max_length=1024,  # Increase if necessary to accommodate long sequences
        return_tensors='pt'
    )

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Prepare DataLoader
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8)

# Training setup (example)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Learning rate scheduler
num_epochs = 4
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# The model, optimizer and data loaders are passed to the Accelerator instance
train_dataloader, eval_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer, lr_scheduler
)

print("Begin Training...")

progress_bar = tqdm(range(num_training_steps))

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

print("Training completed.")

print("\nSaving model weights after training")

model_save_path = "finetuned_phi4.pth"
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to: {model_save_path}")