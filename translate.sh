#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=TranslateMissing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=slurm_output_%A.out

# Activate your environment
source venv/bin/activate

python data/eb-nerd/ebnerd_large/ebnerd_translate.py 