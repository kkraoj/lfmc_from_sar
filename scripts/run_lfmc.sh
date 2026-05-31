#!/bin/bash
#SBATCH --job-name=lfmc_map
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --output=$HOME/lfmc_logs/%j.out
#SBATCH --error=$HOME/lfmc_logs/%j.err

export SHERLOCK=1
source /oak/stanford/groups/konings/projects/rao_2020/code/env/activate_lfmc.sh
export PATH=/oak/stanford/groups/konings/projects/rao_2020/code/env/envs/lfmc/bin:$PATH

mkdir -p $HOME/lfmc_logs /oak/stanford/groups/konings/projects/rao_2020/data/lfmc_maps

cd /oak/stanford/groups/konings/projects/rao_2020/code/lfmc_from_sar/scripts
if [ -n "$LFMC_DATE" ]; then
    python -u make_map_features_and_predict.py --date "$LFMC_DATE"
else
    python -u make_map_features_and_predict.py --year "$LFMC_YEAR"
fi
