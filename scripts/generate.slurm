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

srun python -u -m cse599o_basics.generate \
  --checkpoint-load-path checkpoints/checkpoint_lr_schedule_1.pt \
  --prompt-path data/prompt_text.txt \
  --dim 512 \
  --n-layers 4 \
  --n-heads 16 \
  --rope-theta 10000 \
  --ffn-dim 1344 \
  --context-length 256 \
  --temperature 1.1 \
  --top-p 0.9
status=$?
echo "[INFO] Job finished at $(date) with status $status"
exit $status
