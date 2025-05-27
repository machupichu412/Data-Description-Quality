import json
import re

INPUT_FILE = "/Users/louisciccone/micro/Data-Description-Quality/data/inference_results.json"
OUTPUT_FILE = "cleaned_outputs.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

cleaned_outputs = []

for row in data:
    index = row.get("index")
    output_text = row.get("output", "")
    prompt_text = row.get("prompt", "")

    # Clean up escape sequences
    output_text = output_text.replace("\\n", "\n").replace('\\"', '"')
    prompt_text = prompt_text.replace("\\n", "\n").replace('\\"', '"')

    # Extract Description from prompt
    desc_match = re.search(r"<\|user\|>Description:\s*(.*?)<\|end\|>", prompt_text, re.DOTALL)

    # Extract last Reason → Decision pair from output
    matches = re.findall(r"Reason:(.*?)Decision:\s*(Pass|Fail)", output_text, re.DOTALL)

    if matches and desc_match:
        reasoning, decision = matches[-1]
        description = desc_match.group(1).strip()
        cleaned_outputs.append({
            "index": index,
            "description": description,
            "decision": decision.strip(),
            "full_reasoning": f"Reason:{reasoning.strip()}\nDecision: {decision.strip()}"
        })
    else:
        print(f"⚠️ Couldn't extract data at index {index}")

# Save to file
with open(OUTPUT_FILE, "w") as f:
    json.dump(cleaned_outputs, f, indent=2)

print(f"\n✅ Cleaned {len(cleaned_outputs)} outputs.")
