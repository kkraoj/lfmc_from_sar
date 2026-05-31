# LFMC from SAR — Sherlock Setup Notes

## Repo
Cloned from https://github.com/kkraoj/lfmc_from_sar.git  
Location: `$SCRATCH/lfmc_from_sar/` (`/scratch/users/kkrao/lfmc_from_sar/`)

## What this repo does
Predicts Live Fuel Moisture Content (LFMC) by running a pre-trained LSTM model over
gridded inputs combining Landsat 8 optical bands and Sentinel-1 SAR backscatter.
For a given year, it iterates all dates (monthly, 1st and 15th) and produces a GeoTIFF
map for any date where all 4 lag months of input data are available.

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

## Earth Engine authentication
- GCP project: `project-3af726f4-b7ec-4b39-ae4` (My First Project)
- EE account: `rsegcal@gmail.com` (ask a group member for the password)

Each group member authenticates once on Sherlock. Only needs to be done once per user.

**Step 1 — activate the shared env:**
```bash
source /oak/stanford/groups/konings/projects/rao_2020/code/env/activate_lfmc.sh
```

**Step 2 — authenticate:**
```bash
python -c "import ee; ee.Authenticate(auth_mode='notebook', force=True)" 2>/dev/null
```
A URL will be printed. Open it in a browser, sign in with `rsegcal@gmail.com`, and paste the authorization code back into the terminal.

Credentials are saved to `~/.config/earthengine/credentials` and persist across sessions — you won't need to do this again unless they expire.

## Data paths (Sherlock)
All data lives under `$SCRATCH/vwc_from_radar/data/` (set by `SHERLOCK=1` in `dirs.py`):

| File | Size | Purpose |
|------|------|---------|
| `data/encoder.pkl` | ~332 B | LabelEncoder for forest_cover classes |
| `data/scaler.pkl` | ~5 KB | MinMaxScaler (pickled with sklearn 0.19.1) |
| `data/map/static_features_p36_250m_latlon_float32` | 3.8 GB | Static pixel features (elevation, soil, canopy, etc.) |
| `data/map/map_lat_lon_p36_250m_latlon_float32` | 596 MB | Lat/lon grid for output raster reconstruction |
| `lfmc_from_sar/codes/model_checkpoint/LSTM/quality_pure+all_same_28_may_2019_res_1M_gap_3M_site_split_raw_ratios.hdf5` | ~71 KB | Trained LSTM weights (intentionally small model) |
| `data/map/dynamic_maps/inputs_250m/YYYY-MM-DD_cloudsnowfree_l8.tif` | ~3 GB each | Landsat 8 optical inputs (5 bands) |
| `data/map/dynamic_maps/inputs_250m/YYYY-MM-DD_sar.tif` | ~0.8–1 GB each | SAR inputs (vh, vv bands) |

## Running inference

For a single date:
```bash
LFMC_DATE=2025-04-15 sbatch /oak/stanford/groups/konings/projects/rao_2020/code/lfmc_from_sar/scripts/run_lfmc.sh
```

For a full year (iterates all 1st/15th dates):
```bash
LFMC_YEAR=2025 sbatch /oak/stanford/groups/konings/projects/rao_2020/code/lfmc_from_sar/scripts/run_lfmc.sh
```

Output: `/oak/stanford/groups/konings/projects/rao_2020/data/lfmc_maps/lfmc_map_YYYY-MM-DD.tif`

Only dates where all 4 lag months of inputs exist will produce a map.

**Resource requirements:** 256 GB RAM, ~45 min total runtime (8 min loading,
~10 min bucket processing, ~20 min LSTM prediction, ~5 min saving).
128 GB is sufficient for Jan–Aug 2025 dates (smaller SAR files ~830 MB), but
Sep 2025 onward uses ~1.5 GB SAR files per lag and OOM-kills at 128 GB.

## Bugs fixed (May 2025)

All fixes are in `scripts/make_map_features_and_predict.py` and `scripts/run_lfmc.sh`.

### 1. Wrong `cd` path in `run_lfmc.sh`
Scripts live in `scripts/`, not `codes/scripts/`.  
**Fix:** `cd $SCRATCH/lfmc_from_sar/scripts`

### 2. Memory too low in `run_lfmc.sh`
Original `--mem=64G` was OOM-killed during bucket reassembly (~80M rows × 120 float32 cols ≈ 38 GB for latlon alone, plus prediction arrays).  
**Fix:** `#SBATCH --mem=128G`

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
- `$SCRATCH` is purged after 90 days of no access — copy important outputs to `$OAK` or `$HOME`
- SLURM logs go to `$HOME/lfmc_logs/%j.{out,err}` (each user's home — automatically created on first run)
- The overflow warnings (`RuntimeWarning: overflow encountered in cast`) in stderr are expected and harmless — they come from float16 arithmetic before the float64 cast
- The sklearn/TF version warnings on startup are also harmless for inference
