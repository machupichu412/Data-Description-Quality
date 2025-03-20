#!/bin/bash
#SBATCH --job-name=llama_finetune
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --mem=180GB
#SBATCH --time=4:00:00
#SBATCH --output=finetune_llama.log

module load python/3.9.12
module load cuda/11.7.1
module load gcc/10.3.0           # Load GCC prerequisite
module load openmpi/5.0.3-cuda   # Now load the CUDA-enabled OpenMPI

export CUDA_HOME=/sw/pkgs/arc/cuda/11.7.1
export MPICC=$(which mpicc)
which nvcc
which mpicc

source ~/venv_llama/bin/activate
export TOKENIZERS_PARALLELISM=false

pip install mpi4py




python training_model.py

