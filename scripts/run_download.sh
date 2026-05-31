#!/bin/bash
#SBATCH --job-name=lfmc_download
#SBATCH --partition=normal
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=$HOME/lfmc_logs/%j.out
#SBATCH --error=$HOME/lfmc_logs/%j.err

export SHERLOCK=1
source /oak/stanford/groups/konings/projects/rao_2020/code/env/activate_lfmc.sh
export PATH=/oak/stanford/groups/konings/projects/rao_2020/code/env/envs/lfmc/bin:$PATH



mkdir -p $HOME/lfmc_logs
cd /oak/stanford/groups/konings/projects/rao_2020/code/lfmc_from_sar/scripts
python download_from_gcs.py "$@"
