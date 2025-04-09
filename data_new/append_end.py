import json

input_path = "train_ALL_v2.jsonl"
output_path = "train_ALL_v2_end.jsonl"

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        example = json.loads(line.strip())
        if "decision" in example and not example["decision"].endswith("<|end|>"):
            example["decision"] = example["decision"].strip() + " <|end|>"
        outfile.write(json.dumps(example) + "\n")

print(f"✅ Done. Saved updated file to: {output_path}")
