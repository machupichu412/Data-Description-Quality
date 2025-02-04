import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline

# check processing power
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load dataset
df = pd.read_csv("data.csv")  

# classification labels
df["label"] = df["label"].apply(lambda x: 1 if x == "positive" else 0)

# convert to hugging face format
dataset = Dataset.from_pandas(df)

# test split
dataset = dataset.train_test_split(test_size=0.2)

# load llama 2
model_name = "meta-llama/Llama-2-7b-hf"  # can change this based on computing power
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# tokenize data
def preprocess_function(examples):
    return tokenizer(examples["combined_text"], truncation=True, padding=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# training settings
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,  # change based on computing power
    per_device_eval_batch_size=2,
    num_train_epochs=3,  
    weight_decay=0.01,
    logging_dir="./logs",
)

# train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

# save new model
trainer.save_model("fine_tuned_llama")
tokenizer.save_pretrained("fine_tuned_llama")

# load it as a classifier 
classifier = pipeline("text-classification", model="fine_tuned_llama", device=0 if torch.cuda.is_available() else -1)

# results
test_comment = "Entity: This entity stores customer transactions. | Attribute: Transaction Amount field stores monetary values. | Service: Used for financial reporting."
result = classifier(test_comment)
print(f"Classification Result: {result}")
