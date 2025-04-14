#!/bin/bash
#SBATCH --job-name=llama_infer_gpu
#SBATCH --account=jinseokk0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=llama_infer_gpu.log

# Load necessary modules
module load python/3.9.12 cuda/11.8

# Activate your virtual environment
source ~/venv_llama/bin/activate

# Avoid tokenizer multiprocessing issues
export TOKENIZERS_PARALLELISM=false

# Run inference
python test_model.py
