#!/bin/bash
#SBATCH --job-name=lfmc_pipeline
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/users/kkrao/lfmc_logs/pipeline_%j.out
#SBATCH --error=/scratch/users/kkrao/lfmc_logs/pipeline_%j.err
#
# Full LFMC pipeline: GEE export → GCS download → L8 merge → inference.
#
# Usage:
#   sbatch run_lfmc_pipeline.sh 2025-05-15   # explicit target date (1st or 15th)
#   sbatch run_lfmc_pipeline.sh              # defaults to latest 1st/15th of today
#
# TARGET_DATE is the as-of (end) date for the LFMC map. The script resolves the
# 4 monthly lag dates needed by inference and skips any that are already present
# locally (idempotent). Already-complete LFMC output maps are also skipped.

set -euo pipefail

SCRIPTS=/scratch/users/kkrao/lfmc_from_sar/scripts
INPUTS=/scratch/users/kkrao/vwc_from_radar/data/map/dynamic_maps/inputs_250m
LFMC_OUT=/scratch/users/kkrao/vwc_from_radar/data/map/dynamic_maps/lfmc

export SHERLOCK=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lfmc

mkdir -p "$INPUTS" "$LFMC_OUT" /scratch/users/kkrao/lfmc_logs

# ── Resolve TARGET_DATE ──────────────────────────────────────────────────────
# If not provided, default to the most recent 1st or 15th on or before today.
if [ -n "${1:-}" ]; then
    TARGET_DATE="$1"
else
    TARGET_DATE=$(python3 -c "
from datetime import date
today = date.today()
d = today.replace(day=15) if today.day >= 15 else today.replace(day=1)
print(d.strftime('%Y-%m-%d'))
")
    echo "[INFO] No date given — defaulting to latest fortnight: $TARGET_DATE"
fi

# Basic validation: must be YYYY-MM-DD with day 01 or 15
if ! [[ "$TARGET_DATE" =~ ^[0-9]{4}-[0-9]{2}-(01|15)$ ]]; then
    echo "[ERROR] TARGET_DATE must be YYYY-MM-DD with day 01 or 15, got: $TARGET_DATE" >&2
    exit 1
fi

# ── Compute lag dates (t-3, t-2, t-1, t) ───────────────────────────────────
LAG_DATES=$(python3 -c "
import pandas as pd
target = pd.to_datetime('$TARGET_DATE')
dates = [(target - pd.DateOffset(months=lag)).strftime('%Y-%m-%d') for lag in range(3, -1, -1)]
print(' '.join(dates))
")

echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[INFO] LFMC pipeline — target: $TARGET_DATE"
echo "[INFO] Lag dates: $LAG_DATES"
echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Short-circuit if LFMC output already exists ─────────────────────
LFMC_MAP="$LFMC_OUT/lfmc_map_${TARGET_DATE}.tif"
if [ -f "$LFMC_MAP" ]; then
    echo "[INFO] LFMC map already exists: $LFMC_MAP — nothing to do."
    exit 0
fi

# ── Step 2: Export from GEE (only dates missing locally) ────────────────────
NEED_EXPORT=()
for d in $LAG_DATES; do
    # "Present" means at least one l8 file AND at least one sar file exist
    # (covers both merged single-file and multi-chunk cases)
    l8_count=$(ls "$INPUTS"/${d}_cloudsnowfree_l8*.tif 2>/dev/null | wc -l)
    sar_count=$(ls "$INPUTS"/${d}_sar*.tif 2>/dev/null | wc -l)
    if [ "$l8_count" -gt 0 ] && [ "$sar_count" -gt 0 ]; then
        echo "[INFO] $d — inputs present locally, skipping GEE export."
    else
        echo "[INFO] $d — inputs missing (l8=$l8_count, sar=$sar_count), queuing for export."
        NEED_EXPORT+=("$d")
    fi
done

if [ "${#NEED_EXPORT[@]}" -gt 0 ]; then
    echo "[INFO] Exporting ${#NEED_EXPORT[@]} date(s) from GEE: ${NEED_EXPORT[*]}"
    python "$SCRIPTS/export_gee_maps.py" --dates "${NEED_EXPORT[@]}" --poll
else
    echo "[INFO] All lag dates present — skipping GEE export."
fi

# ── Step 3: Download from GCS (skips files that match local size) ────────────
YEAR_ARGS=$(python3 -c "
dates = '$LAG_DATES'.split()
years = sorted(set(d[:4] for d in dates))
print(' '.join('--year ' + y for y in years))
")
echo "[INFO] Downloading from GCS ($YEAR_ARGS)..."
python "$SCRIPTS/download_from_gcs.py" $YEAR_ARGS

# ── Step 4: Merge chunked L8 tiles (no-op if no chunks present) ─────────────
echo "[INFO] Merging any split L8 tiles..."
python "$SCRIPTS/merge_l8_tiles.py"

# ── Step 5: LFMC inference ───────────────────────────────────────────────────
echo "[INFO] Running LFMC inference for $TARGET_DATE..."
python -u "$SCRIPTS/make_map_features_and_predict.py" --date "$TARGET_DATE"

echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[INFO] Done. Output: $LFMC_MAP"
echo "[INFO] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
