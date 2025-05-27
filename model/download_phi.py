import os
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token = os.getenv("HF_TOKEN")  # ✅ Get the token from the environment

# Local save path
local_model_dir = os.path.expanduser("~/phi4_mini_instruct_local")
os.makedirs(local_model_dir, exist_ok=True)

# Phi-4-mini-instruct repo on Hugging Face
repo_id = "microsoft/Phi-4-mini-instruct"

# Load and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
tokenizer.save_pretrained(local_model_dir)

# Load and save model
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    torch_dtype="auto",
    device_map="auto",
    token=hf_token,
    trust_remote_code=True,  # recommended for Phi-4
)
model.save_pretrained(local_model_dir)

print(f"✅ Phi-4-mini-instruct model and tokenizer saved to {local_model_dir}!")

