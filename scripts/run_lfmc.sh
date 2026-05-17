#!/bin/bash
#SBATCH --job-name=lfmc_map
#SBATCH --partition=normal
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/users/kkrao/lfmc_logs/%j.out
#SBATCH --error=/scratch/users/kkrao/lfmc_logs/%j.err

export SHERLOCK=1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lfmc

mkdir -p /scratch/users/kkrao/vwc_from_radar/data/map/dynamic_maps/lfmc

cd /scratch/users/kkrao/lfmc_from_sar/codes/scripts
python make_map_features_and_predict.py --year "$LFMC_YEAR"
