#!/bin/bash
#SBATCH --job-name=train_llava        # Job name
#SBATCH --output=output_%j.txt       # Output file (%j will be replaced by job ID)
#SBATCH --error=error_%j.txt         # Error file
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks per node (usually 1 per GPU)
#SBATCH --gpus=8                     # Number of GPUs
#SBATCH --time=48:00:00              # Time limit (48 hours)
#SBATCH --partition=gpu              # GPU partition (if required by your system)
#SBATCH --mem=256G                    # Memory required

# Activate your environment if needed
source activate llava

# Run your Python or other executable script
bash scripts/train/pretrain_clip_llava15_fast.sh
