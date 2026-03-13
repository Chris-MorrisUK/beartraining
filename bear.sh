#!/bin/bash

#SBATCH --ntasks=32
#SBATCH --time=3:00:00
#SBATCH --qos=bbgpu
#SBATCH --account=morriscz-sch-plan-digit
#SBATCH --gres=gpu:a100:2


set -e

module purge; module load bluebear
module load bear-apps/2024a
module load Python/3.12.3-GCCcore-13.3.0

cd /rds/homes/m/morriscz/beartraining
#should handle the venv as well as the dependencies
uv sync
#these parameter control the training
export training_data_path="/rds/homes/m/morriscz/custom/custom.yaml"
#these have to be significantly smalleron my local machine to avoid OOM errors, but I can increase them on bluebear
export BATCH_SIZE=4
export EPOCHS=100
uv run train.py

