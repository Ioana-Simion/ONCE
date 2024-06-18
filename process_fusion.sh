#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Fusion
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=process_fusion_%A.out

cd $HOME/ONCE

# Activate your environment
source venv/bin/activate

pip install unitok==3.5.2

python fusion.py