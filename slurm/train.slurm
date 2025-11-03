#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=puyuan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --chdir=/homes/iws/puyuan/cse599o/assignment1-basics
#SBATCH --export=ALL
#SBATCH --output=log/puyuan-train-slurm-%j.out
#SBATCH --error=log/puyuan-train-slurm-%j.err

set -euo pipefail

echo "[INFO] Job started on $(hostname) at $(date)"

mkdir -p checkpoints

srun python -u scripts/train.py \
  --batch-size 32 \
  --train-data-path data/TinyStoriesV2-GPT4-train.bin \
  --valid-data-path data/TinyStoriesV2-GPT4-valid.bin \
  --checkpoint-load-path checkpoints/checkpoint.pt \
  --checkpoint-iters 500 \
  --checkpoint-save-path checkpoints/checkpoint.pt \
  --loss-log-path log/loss.csv \
  --dim 512 \
  --n-layers 4 \
  --n-heads 16 \
  --rope-theta 10000 \
  --ffn-dim 1344 \
  --context-length 256 \
  --iters 5000 \
  --optim adamw \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --beta1 0.9 \
  --beta2 0.95 \
  --eps 1e-8 \
  --use-gradient-clipping True \
  --max-norm-gradient-clipping 1.0 \
  --eps-gradient-clipping 1e-6 \
  --use-learning-rate-schedule True \
  --lr-max 1e-3 \
  --lr-min 2e-5 \
  --warm-up-iters 300 \
  --cosine-annealing-iters 5000
status=$?
echo "[INFO] Job finished at $(date) with status $status"
exit $status
