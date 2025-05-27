import os
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token = os.getenv("HF_TOKEN")  # ✅ Get the token from the environment

local_model_dir = os.path.expanduser("~/llama3_8B_local")
os.makedirs(local_model_dir, exist_ok=True)

repo_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
tokenizer.save_pretrained(local_model_dir)

model = AutoModelForCausalLM.from_pretrained(
    repo_id, 
    torch_dtype="auto", 
    device_map="auto",
    token=hf_token
)
model.save_pretrained(local_model_dir)

print(f"✅ Model and tokenizer successfully downloaded to {local_model_dir}!")

