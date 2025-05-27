#!/bin/bash
#SBATCH --job-name=finetune-phi
#SBATCH --account=jinseokk0
#SBATCH --output=finetune_phi.out
#SBATCH --error=finetune_phi.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1        # Request a GPU if available for faster computation
#SBATCH --time=4:00:00     # Set an appropriate walltime
#SBATCH --mem=64GB           # Adjust based on your needs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mattyeh@umich.edu

# Load necessary modules if required, e.g., CUDA if you're using GPU
# module load cuda/11.3  # Example for CUDA; change to the version used on your system

# Activate your Python virtual environment
source phi_venv/bin/activate
pip install datasets

# Run your fine-tuning script
python3 finetuning_phi4.py