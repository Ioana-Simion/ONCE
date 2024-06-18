#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Worker
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=worker_prep_%A.out

cd $HOME/ONCE

# Activate your environment
source venv/bin/activate

python worker.py \
    --embed config/embed/ebnerd-embed.yaml \
    --model config/model/llm/ebnerd-model.yaml \
    --exp config/exp/ebnerd-exp-prep.yaml \
    --data config/data/ebnerd-data.yaml \
    --version small \
    --llm_ver 7b \
    --hidden_size 64 \
    --layer 0 \
    --lora 0 \
    --fast_eval 0 \
    --embed_hidden_size 4096 \
    --page_size 8 \