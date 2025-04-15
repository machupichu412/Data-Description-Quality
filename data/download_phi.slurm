#!/bin/bash
#SBATCH --job-name=phi4_download
#SBATCH --account=jinseokk0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=phi_download.log

# ✅ Load required modules
module load python/3.9.12
module load cuda/11.7.1

# ✅ Activate your virtual environment
source ~/venv_llama/bin/activate

# ✅ Set Hugging Face token (or use environment variable)
export HF_TOKEN=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


# ✅ Run your Python download script
python /home/cicconel/llama_models/download_phi.py

