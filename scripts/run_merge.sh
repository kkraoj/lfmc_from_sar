#!/bin/bash
#SBATCH --job-name=lfmc_merge
#SBATCH --partition=normal
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/users/kkrao/lfmc_logs/%j.out
#SBATCH --error=/scratch/users/kkrao/lfmc_logs/%j.err

export SHERLOCK=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lfmc

cd /scratch/users/kkrao/lfmc_from_sar/scripts
python merge_l8_tiles.py
