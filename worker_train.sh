#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=WorkerTrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=worker_train_%A.out

cd $HOME/ONCE

# Activate your environment
source venv/bin/activate

# works with page_size both 8 and 16
python worker.py \
    --embed config/embed/ebnerd-embed.yaml \
    --model config/model/llm/ebnerd-model.yaml \
    --exp config/exp/ebnerd-exp-train.yaml \
    --data config/data/ebnerd-data.yaml \
    --version small \
    --llm_ver 7b \
    --hidden_size 64 \
    --layer 30 \
    --lora 1 \
    --fast_eval 0 \
    --embed_hidden_size 4096 \
    --page_size 16 \
    --batch_size 16 \