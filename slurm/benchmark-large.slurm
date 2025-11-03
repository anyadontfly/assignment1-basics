#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=puyuan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --chdir=/homes/iws/puyuan/cse599o/assignment1-basics
#SBATCH --export=ALL
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

CONTEXT_LENGTH=${1:-128}
PROFILE_RANGE=${2:-f}

OUTPUT_FILE="log/puyuan-benchmark-large-${CONTEXT_LENGTH}-${PROFILE_RANGE}-slurm-${SLURM_JOB_ID}.out"
ERROR_FILE="log/puyuan-benchmark-large-${CONTEXT_LENGTH}-${PROFILE_RANGE}-slurm-${SLURM_JOB_ID}.err"

exec 1> "${OUTPUT_FILE}"
exec 2> "${ERROR_FILE}"

echo "Context Length: ${CONTEXT_LENGTH}"
echo "Profile Range: ${PROFILE_RANGE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output: ${OUTPUT_FILE}"
echo "Error: ${ERROR_FILE}"
echo ""

srun nsys profile -t nvtx,cuda -o "nsys-rep/large-${CONTEXT_LENGTH}-${PROFILE_RANGE}" \
  python -u scripts/benchmark.py \
  --batch-size 4 \
  --dim 1280 \
  --n-layers 36 \
  --n-heads 20 \
  --rope-theta 10000 \
  --ffn-dim 5120 \
  --context-length ${CONTEXT_LENGTH} \
  --iters-warmup 5 \
  --iters-benchmark 10 \
  --optim adamw \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --beta1 0.9 \
  --beta2 0.95 \
  --eps 1e-8 \
  --vocab-size 50257 \
  --profile-range ${PROFILE_RANGE} \
  --mem
