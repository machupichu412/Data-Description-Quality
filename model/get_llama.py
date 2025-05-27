import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Define Hugging Face Repo ID
repo_id = "meta-llama/Llama-3.1-8B"

# ✅ Define absolute local directory
local_model_dir = os.path.abspath("llama3_8B_local")  

# ✅ Step 1: Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)
tokenizer.save_pretrained(local_model_dir)

# ✅ Step 2: Download and save the model
model = AutoModelForCausalLM.from_pretrained(repo_id)
model.save_pretrained(local_model_dir)

print(f"✅ Model and tokenizer successfully downloaded to {local_model_dir}!")
