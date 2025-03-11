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

# ✅ Check processing power
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#  Load dataset
data = []
with open("train.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame([entry["messages"][-2]["content"] for entry in data], columns=["TechnicalDescription"])

#  Categorization function
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

#  Apply categorization
df["failure_reason"] = df["TechnicalDescription"].apply(categorize_failure)

#  Save categorized failures to a new file
df.to_csv("categorized_failures.csv", index=False)

#  Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

#  Split dataset
dataset = dataset.train_test_split(test_size=0.2)

#  Load Local LLaMA Model
local_model_dir = "/home/cicconel/llama3_8B_local"
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForSequenceClassification.from_pretrained(
    local_model_dir,
    num_labels=2,  # Adjust for your task
    torch_dtype=torch.float16,  # Use FP16 to reduce memory usage
    device_map="auto",  # Offload automatically
)

#  Tokenization function
def preprocess_function(examples):
    return tokenizer(
        examples["TechnicalDescription"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )



tokenized_datasets = dataset.map(preprocess_function, batched=True)

#  Training settings (adjusted for memory limits)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,  # Reduced batch size to prevent OOM
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True if torch.cuda.is_available() else False,  # Use FP16 only on GPUs
)

#  Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

#  Train model
trainer.train()

#  Save fine-tuned model
trainer.save_model("fine_tuned_llama")
tokenizer.save_pretrained("fine_tuned_llama")

#  Test inference
test_comment = "This Attribute has a static value - <3001> decided by commercial business."
inputs = tokenizer(test_comment, return_tensors="pt").to(device)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1).item()

classification = "Pass" if predictions == 0 else "Fail"
reason = "N/A" if predictions == 0 else categorize_failure(test_comment)

print(f"Classification Result: {classification}, Reason: {reason}")

