#!/bin/bash
#SBATCH --job-name=llama_download
#SBATCH --account=jinseokk0   # Change to your ARC account name
#SBATCH --partition=standard          # Use standard partition
#SBATCH --mem=64G                     # Request enough memory (adjust if needed)
#SBATCH --time=04:00:00               # Set time limit (adjust if needed)
#SBATCH --output=llama_download.log   # Save output logs

# ✅ Load Python module
module load python/3.9.12  # Use the correct version available on Great Lakes

# ✅ Activate your virtual environment
source ~/venv_llama/bin/activate

# ✅ Run the Python script to download the model
deepspeed --num_gpus=4 training_model.py

