import pandas as pd
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    pipeline
)

#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
# Check processing power
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
data = []
with open("train.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame([entry["messages"][-2]["content"] for entry in data], columns=["TechnicalDescription"])

# Categorization function
def categorize_failure(description):
    if "direct mapping" in description.lower() or "mapping" in description.lower():
        return "Missing source info"
    elif "transformation" in description.lower() or "logic" in description.lower():
        return "Transformation logic"
    elif "unclear" in description.lower() or "not clear" in description.lower() or "difficult to understand" in description.lower():
        return "Unclear sentence"
    elif "join" in description.lower() or "conditions" in description.lower():
        return "Conditions of join"
    else:
        return "Uncategorized"

# Apply categorization
df["failure_reason"] = df["TechnicalDescription"].apply(categorize_failure)

# Save categorized failures to a new file
df.to_csv("categorized_failures.csv", index=False)

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Split dataset
dataset = dataset.train_test_split(test_size=0.2)

# Load LLaMA model and tokenizer
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.1-8B").to(device)

# Tokenization function
def preprocess_function(examples):
    return tokenizer(examples["TechnicalDescription"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

#~/llama_models/

# Training settings
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Train model
trainer.train()

# Save fine-tuned model
trainer.save_model("fine_tuned_llama")
tokenizer.save_pretrained("fine_tuned_llama")

# Test inference
test_comment = "This Attribute has a static value - <3001> decided by commercial business."
inputs = tokenizer(test_comment, return_tensors="pt").to(device)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1).item()

if predictions == 0:
    classification = "Pass"
    reason = "N/A"
else:
    classification = "Fail"
    reason = categorize_failure(test_comment)

print(f"Classification Result: {classification}, Reason: {reason}")
