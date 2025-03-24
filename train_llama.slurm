#!/bin/bash
#SBATCH --job-name=llama_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2              # Most nodes support 2
#SBATCH --mem=120GB
#SBATCH --time=4:00:00
#SBATCH --output=finetune_llama.log
#SBATCH --account=jinseokk0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

module load python/3.9.12
module load cuda/11.7.1
module load gcc/10.3.0
module load openmpi/5.0.3-cuda

source ~/venv_llama/bin/activate
export TOKENIZERS_PARALLELISM=false

# Make sure mpi4py is installed
pip install --quiet mpi4py

# Adjust DeepSpeed launch to match number of GPUs
deepspeed --num_gpus=2 training_model.py
