#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=WorkerTrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=worker_train_1_epoch_%A.out

cd $HOME/ONCE

# Activate your environment
source venv/bin/activate

# pip install scikit-learn==1.1.2
# pip install numpy==1.23.2

python worker.py \
    --embed config/embed/ebnerd-embed.yaml \
    --model config/model/llm/ebnerd-model.yaml \
    --exp config/exp/ebnerd-exp-train.yaml \
    --data config/data/ebnerd-data.yaml \
    --version small \
    --llm_ver 7b \
    --hidden_size 64 \
    --layer 1 \
    --lora 0 \
    --fast_eval 0 \
    --embed_hidden_size 4096 \
    --page_size 8 \