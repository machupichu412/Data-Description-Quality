#!/bin/bash
#SBATCH --job-name=phi_finetune
#SBATCH --account=jinseokk0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                   # Use 1 or 2 GPUs; most nodes don’t support 3
#SBATCH --mem=120GB
#SBATCH --time=4:00:00
#SBATCH --output=phi_finetune.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# Load modules
module load python/3.9.12
module load cuda/11.7.1
module load gcc/10.3.0
module load openmpi/5.0.3-cuda

# Activate virtual environment
source ~/venv_llama/bin/activate

# Avoid CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Ensure required packages are present
pip install --quiet mpi4py charset-normalizer

# Launch DeepSpeed training
deepspeed --num_gpus=1 training_model.py