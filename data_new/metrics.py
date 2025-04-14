import json
from sklearn.metrics import classification_report

# Load ground truth data
with open("test_ALL_v2.jsonl", "r") as f:
    gold_data = [json.loads(line) for line in f]

# Load inference results
with open("inference_results.json", "r") as f:
    pred_data = json.load(f)

# Extract true and predicted labels
true_labels = []
pred_labels = []

for gold, pred in zip(gold_data, pred_data):
    true_decision = gold["decision"].strip().lower()
    true_label = 1 if true_decision == "pass" else 0

    model_output = pred["output"].lower()
    if "decision: pass" in model_output:
        pred_label = 1
    elif "decision: fail" in model_output:
        pred_label = 0
    else:
        # If no decision detected, count as incorrect
        pred_label = -1

    true_labels.append(true_label)
    pred_labels.append(pred_label)

# Filter out undecided predictions
filtered = [(t, p) for t, p in zip(true_labels, pred_labels) if p != -1]
true_labels, pred_labels = zip(*filtered)

# Print metrics
print(classification_report(true_labels, pred_labels, target_names=["Fail", "Pass"]))
