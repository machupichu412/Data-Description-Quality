import json
from sklearn.metrics import classification_report

# Load labeled predictions
with open("inference_results_labeled.json", "r") as f:
    data = json.load(f)

# Extract true and predicted labels
true_labels = []
pred_labels = []

for row in data:
    output = row["output"].lower().strip().replace("<|end|>", "").strip()

    # Use the attached label if present
    label = row.get("label")
    if label is None:
        continue  # skip if no label
    
    true_labels.append(int(label))
    pred_labels.append(1 if output == "pass" else 0)

# Compute and display metrics
print(classification_report(true_labels, pred_labels, target_names=["Fail", "Pass"]))
