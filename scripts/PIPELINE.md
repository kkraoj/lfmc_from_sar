# LFMC Pipeline — `run_lfmc_pipeline.sh`

End-to-end script that produces one LFMC map for a target date. Runs as a
single SLURM job: GEE export → GCS download → L8 tile merge → inference.
Every step is idempotent — re-running for a date that's already complete is
safe and fast.

## Usage

```bash
# Explicit target date (must be 1st or 15th of month):
sbatch run_lfmc_pipeline.sh 2025-05-15

# No argument — defaults to the most recent 1st or 15th on or before today:
sbatch run_lfmc_pipeline.sh
```

The **target date** is the as-of (end) date of the LFMC map. Inference
requires 4 monthly lag inputs: `t`, `t-1mo`, `t-2mo`, `t-3mo`. For a target
of `2025-05-15` those are `2025-05-15`, `2025-04-15`, `2025-03-15`,
`2025-02-15`. The earliest raw satellite imagery used is ~3 months before the
oldest lag (`≈ 2024-11-15` in this example).

For a cron job, run on the 1st and 16th of each month (the day after a new
fortnight date lands, giving GEE time to ingest fresh imagery):

```cron
0 6 1,16 * * sbatch /scratch/users/kkrao/lfmc_from_sar/scripts/run_lfmc_pipeline.sh
```

---

## Directories and artifacts

### GCS bucket (upstream source)

| Path | Description |
|------|-------------|
| `gs://lfmc-inputs/inputs_250m/YYYY-MM-DD_cloudsnowfree_l8*.tif` | Landsat 8 composite (5 bands: B2–B6) exported from GEE |
| `gs://lfmc-inputs/inputs_250m/YYYY-MM-DD_sar*.tif` | Sentinel-1 SAR composite (VH, VV) exported from GEE |

GEE splits large exports into numbered chunks (`…0000000000-0000000000.tif`).
`merge_l8_tiles.py` stitches these into single files after download.

### Local inputs directory

```
$SCRATCH/vwc_from_radar/data/map/dynamic_maps/inputs_250m/
```

| File pattern | Size | Description |
|--------------|------|-------------|
| `YYYY-MM-DD_cloudsnowfree_l8.tif` | ~3 GB | Merged Landsat 8 composite |
| `YYYY-MM-DD_sar.tif` | ~0.8–1.5 GB | Merged SAR composite |
| `YYYY-MM-DD_cloudsnowfree_l8NNNNNNNNNn-NNNNNNNNNN.tif` | partial | Pre-merge L8 chunks (deleted after merge) |

The script checks for any file matching `{date}_cloudsnowfree_l8*.tif` and
`{date}_sar*.tif` to decide whether a date is already present. A date is
skipped for GEE export only when **both** l8 and sar files exist.

### Local LFMC output directory

```
$SCRATCH/vwc_from_radar/data/map/dynamic_maps/lfmc/
```

| File | Description |
|------|-------------|
| `lfmc_map_YYYY-MM-DD.tif` | LFMC map for the target date |

If this file already exists when the job starts, the script exits immediately
without doing any work.

### SLURM logs

```
/scratch/users/kkrao/lfmc_logs/pipeline_<JOBID>.out
/scratch/users/kkrao/lfmc_logs/pipeline_<JOBID>.err
```

---

## Step-by-step idempotency

| Step | Skip condition |
|------|---------------|
| Whole job | `lfmc_map_TARGET_DATE.tif` already exists |
| GEE export (per lag date) | Both `{date}_cloudsnowfree_l8*.tif` and `{date}_sar*.tif` exist in local inputs dir |
| GCS download (per file) | Local file exists and byte size matches GCS blob |
| L8 tile merge | No chunk-named files present (no-op) |
| LFMC inference | `lfmc_map_{date}.tif` already exists (handled inside `make_map_features_and_predict.py`) |

---

## Resource requirements

| Resource | Value |
|----------|-------|
| Memory | 256 GB (inference dominates; 128 GB sufficient for pre-Sep 2025 dates) |
| CPUs | 4 |
| Wall time | 12 h (covers ~1 h GEE export + ~1 h download + ~45 min inference) |
| Partition | normal |

---

## Static files required (must already exist)

These are read-only inputs that the pipeline does not create:

| Path | Size | Purpose |
|------|------|---------|
| `$SCRATCH/vwc_from_radar/data/map/map_lat_lon_p36_250m_latlon_float32` | 596 MB | Lat/lon pixel grid for raster reconstruction |
| `$SCRATCH/vwc_from_radar/data/map/static_features_p36_250m_latlon_float32` | 3.8 GB | Static features (elevation, soil, canopy, etc.) |
| `$SCRATCH/vwc_from_radar/data/encoder.pkl` | ~332 B | LabelEncoder for forest cover classes |
| `$SCRATCH/vwc_from_radar/data/scaler.pkl` | ~5 KB | MinMaxScaler for input normalization |
| `$SCRATCH/lfmc_from_sar/codes/model_checkpoint/LSTM/quality_pure+all_same_28_may_2019_res_1M_gap_3M_site_split_raw_ratios.hdf5` | ~71 KB | Trained LSTM weights |

---

## Scripts called

| Script | Role |
|--------|------|
| `export_gee_maps.py` | Queues GEE export tasks and polls until complete |
| `download_from_gcs.py` | Downloads from GCS bucket to local inputs dir |
| `merge_l8_tiles.py` | Merges GEE chunk files into single GeoTIFFs |
| `make_map_features_and_predict.py` | Builds feature array and runs LSTM inference |
