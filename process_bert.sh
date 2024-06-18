#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Processor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=processor_%A.out

cd $HOME/ONCE

# Activate your environment
source venv/bin/activate

python process/eb-nerd/processor_author.py