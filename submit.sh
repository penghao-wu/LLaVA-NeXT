#!/bin/bash
#SBATCH --job-name=train_llava        # Job name
#SBATCH --output=output_%j.txt       # Output file (%j will be replaced by job ID)
#SBATCH --error=error_%j.txt         # Error file
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks per node (usually 1 per GPU)
#SBATCH --gpus=4                     # Number of GPUs
#SBATCH --cpus-per-task=24          # Number of CPU cores per task
#SBATCH --time=35:00:00              # Time limit (48 hours)
#SBATCH --partition=gpu              # GPU partition (if required by your system)
#SBATCH --mem=128G                    # Memory required

export LD_LIBRARY_PATH=/mnt/sfs-common/krhu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/sfs-common/krhu/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:/mnt/sfs-common/krhu/miniconda3/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH
# Activate your environment if needed
source activate llava

# Run your Python or other executable script
bash scripts/train/finetune_clip_llava16_05b.sh
