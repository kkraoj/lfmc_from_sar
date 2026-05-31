# LFMC from SAR — Sherlock Setup Notes

## Repo
Cloned from https://github.com/kkraoj/lfmc_from_sar.git  
Location: `/oak/stanford/groups/konings/projects/rao_2020/code/lfmc_from_sar/`

## What this repo does
Predicts Live Fuel Moisture Content (LFMC) by running a pre-trained LSTM model over
gridded inputs combining Landsat 8 optical bands and Sentinel-1 SAR backscatter.
For a given target date, it uses 4 monthly lag inputs and produces a GeoTIFF map.

## Environment
- Conda env: `lfmc` shared at `/oak/stanford/groups/konings/projects/rao_2020/code/env/envs/lfmc`
- Installed via **micromamba** (not conda — the classic conda solver OOM-kills on login nodes and even on 32 GB compute nodes for this env)
- Key packages: tensorflow 2.20, tf_keras 2.20, pandas 2.x, scikit-learn 1.6, rasterio, matplotlib, seaborn

To activate (works for anyone in the group):
```bash
source /oak/stanford/groups/konings/projects/rao_2020/code/env/activate_lfmc.sh
```

To deactivate:
```bash
micromamba deactivate
```

Optional shortcut — add to `~/.bashrc`:
```bash
alias lfmc="source /oak/stanford/groups/konings/projects/rao_2020/code/env/activate_lfmc.sh"
```

To reinstall from scratch:
```bash
MAMBA_ROOT_PREFIX=/oak/stanford/groups/konings/projects/rao_2020/code/env $HOME/bin/micromamba create -n lfmc \
  -c conda-forge tensorflow pandas scikit-learn rasterio matplotlib seaborn -y
pip install tf_keras
```

## GCP / Earth Engine authentication
- GCP project: `project-3af726f4-b7ec-4b39-ae4`
- Auth method: **service account key** (no manual browser login needed)
- Key file: `/oak/stanford/groups/konings/projects/rao_2020/code/env/gcp_service_account.json` (group-readable, never commit)
- Service account: `lfmc-103@project-3af726f4-b7ec-4b39-ae4.iam.gserviceaccount.com`

The key file is used automatically by all scripts — no setup required for new group members.

## Data paths (Sherlock)
All data lives under `/oak/stanford/groups/konings/projects/rao_2020/data/`:

| Path | Size | Purpose |
|------|------|---------|
| `data/encoder.pkl` | ~332 B | LabelEncoder for forest_cover classes |
| `data/scaler.pkl` | ~5 KB | MinMaxScaler (pickled with sklearn 0.19.1) |
| `data/static_features_p36_250m_latlon_float32` | 3.8 GB | Static pixel features (elevation, soil, canopy, etc.) |
| `data/map_lat_lon_p36_250m_latlon_float32` | 596 MB | Lat/lon grid for output raster reconstruction |
| `code/lfmc_from_sar/model_checkpoint/LSTM/quality_pure+all_same_28_may_2019_res_1M_gap_3M_site_split_raw_ratios.hdf5` | ~71 KB | Trained LSTM weights |
| `data/inputs_250m/YYYY-MM-DD_cloudsnowfree_l8.tif` | ~3 GB each | Landsat 8 optical inputs (5 bands) |
| `data/inputs_250m/YYYY-MM-DD_sar.tif` | ~1.5 GB each | SAR inputs (VH, VV bands) |
| `data/lfmc_maps/lfmc_map_YYYY-MM-DD.tif` | ~274 MB each | Output LFMC maps |

## Running the pipeline

For end-to-end (GEE export → download → merge → inference):
```bash
sbatch /oak/stanford/groups/konings/projects/rao_2020/code/lfmc_from_sar/scripts/run_lfmc_pipeline.sh 2026-06-01
```

For inference only (inputs already present locally):
```bash
LFMC_DATE=2025-04-15 sbatch /oak/stanford/groups/konings/projects/rao_2020/code/lfmc_from_sar/scripts/run_lfmc.sh
```

Output: `/oak/stanford/groups/konings/projects/rao_2020/data/lfmc_maps/lfmc_map_YYYY-MM-DD.tif`

**Resource requirements:** 256 GB RAM, ~90 min total runtime for full pipeline (GEE export ~30 min, download ~15 min, inference ~45 min).

## Bugs fixed (May 2025)

All fixes are in `scripts/make_map_features_and_predict.py` and `scripts/run_lfmc_pipeline.sh`.

### 1. `ls` nonzero exit killing pipeline under `set -euo pipefail`
`ls $INPUTS/${d}_*.tif 2>/dev/null | wc -l` exits 2 when no files match, killing the script.  
**Fix:** Use `find` instead, which always exits 0:
```bash
find "$INPUTS" -maxdepth 1 -name "${d}_cloudsnowfree_l8*.tif" | wc -l
```

### 2. Memory too low
Original `--mem=64G` was OOM-killed. Sep 2025 onward SAR files are ~1.5 GB each (4 lags = 6 GB SAR alone).  
**Fix:** `#SBATCH --mem=256G`

### 3. `scaler.clip` AttributeError (sklearn version mismatch)
`scaler.pkl` was pickled with sklearn 0.19.1; sklearn 1.6.1's `MinMaxScaler.transform()` accesses `self.clip` which didn't exist in 0.19.1.  
**Fix:** After loading the scaler, patch in the missing attribute:
```python
if not hasattr(scaler, 'clip'):
    scaler.clip = False
```

### 4. float16 overflow silently surviving `clip` and `replace`
TIF data is read as `float16` (max ~65504). Ratio features (vh/blue etc.) can be `float16(inf)`. `clip(-1e5, 1e5)` and `replace(inf → 1e5)` were no-ops because storing `1e5` in a float16 column immediately overflows back to `inf`.  
**Fix:** Cast to float64 before sanitizing:
```python
latlon = latlon[cols].astype(np.float64)
latlon.replace([np.inf, -np.inf], [1e5, -1e5], inplace=True)
latlon.dropna(inplace=True)
```

### 5. `.loc` with positional integer slice (pandas 2.0)
`latlon.loc[:,2:]` uses a positional integer slice, which pandas 2.0 disallows on label-indexed DataFrames.  
**Fix:** `latlon.iloc[:,2:]` (replaced in both the bucket path and non-bucket path)

### 6. `DataFrame.append()` removed in pandas 2.0
Bucket reassembly used `latlon.append(...)` which was removed in pandas 2.0.  
**Fix:**
```python
latlon = pd.concat([latlon, pd.read_pickle(...)], ignore_index=True).astype(np.float32)
```

## Notes
- SLURM logs go to `$HOME/lfmc_logs/%j.{out,err}` (each user's home — automatically created on first run)
- The overflow warnings (`RuntimeWarning: overflow encountered in cast`) in stderr are expected and harmless — they come from float16 arithmetic before the float64 cast
- The sklearn/TF version warnings on startup are also harmless for inference
- Never commit `gcp_service_account.json` or any `.json` file (covered by `.gitignore`)
