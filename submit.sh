#!/bin/bash
#SBATCH --job-name=train_llava        # Job name
#SBATCH --output=output_%j.txt       # Output file (%j will be replaced by job ID)
#SBATCH --error=error_%j.txt         # Error file
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks per node (usually 1 per GPU)
#SBATCH --gpus=4                     # Number of GPUs
#SBATCH --cpus-per-task=24          # Number of CPU cores per task
#SBATCH --time=15:00:00              # Time limit (48 hours)
#SBATCH --partition=gpu              # GPU partition (if required by your system)
#SBATCH --mem=128G                    # Memory required

# Activate your environment if needed
source activate llava

# Run your Python or other executable script
bash scripts/train/pretrain_clip_llava15_05b_fast.sh
