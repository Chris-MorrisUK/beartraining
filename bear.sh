#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=240:0
#SBATCH --qos=bbgpu
#SBATCH --account=morriscz-sch-plan-digit
#SBATCH --gres=gpu:a100:2


set -e

module purge; module load bluebear
module load bear-apps/2024a
module load Python/3.12.3-GCCcore-13.3.0

export VENV_DIR="${HOME}/virtual-environments"
export VENV_PATH="${VENV_DIR}/my-virtual-env-${BB_CPU}"

# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
    python3 -m venv --system-site-packages ${VENV_PATH}
fi

# Activate the virtual environment
source ${VENV_PATH}/bin/activate

# Store pip cache in /scratch directory, instead of the default home directory location
#not sure this matters given I'm using uv
PIP_CACHE_DIR="/scratch/${USER}/pip"

uv sync
EXPORT training_data_path="/rds/homes/m/morriscz/custom/custom.yaml"
#these have to be significantly smalleron my local machine to avoid OOM errors, but I can increase them on bluebear
EXPORT BATCH_SIZE=32
EXPORT EPOCHS=100
uv run train.py

