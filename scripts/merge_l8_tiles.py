"""
Merges split Landsat-8 GeoTiff tiles exported from GEE into single files.
GEE splits large exports with suffixes like -0000000000-0000000000.tif.
Run this after downloading from GCS, before running inference.
"""

import glob
import os
import sys
import rasterio
from rasterio.merge import merge

sys.path.insert(0, os.path.dirname(__file__))
from dirs import dir_data
INPUTS_DIR = os.path.join(dir_data, 'map/dynamic_maps/inputs_250m')

tile0_files = sorted(glob.glob(os.path.join(INPUTS_DIR, '*-0000000000-0000000000.tif')))

if not tile0_files:
    print("[INFO] No split tiles found.")
else:
    for tile0 in tile0_files:
        base = tile0.replace('-0000000000-0000000000.tif', '')
        tiles = sorted(glob.glob(base + '-*.tif'))
        merged_path = base + '.tif'
        print(f"[INFO] Merging {[os.path.basename(t) for t in tiles]} -> {os.path.basename(merged_path)}")
        datasets = [rasterio.open(t) for t in tiles]
        # tiles are side by side — same height, combined width
        total_width = sum(ds.width for ds in datasets)
        height = datasets[0].height
        profile = datasets[0].profile.copy()
        profile.update(width=total_width, bigtiff='YES')
        with rasterio.open(merged_path, 'w', **profile) as dst:
            col_offset = 0
            for ds in datasets:
                CHUNK = 256
                for row in range(0, height, CHUNK):
                    actual_height = min(CHUNK, height - row)
                    window = rasterio.windows.Window(0, row, ds.width, actual_height)
                    data = ds.read(window=window)
                    dst.write(data, window=rasterio.windows.Window(col_offset, row, ds.width, actual_height))
                col_offset += ds.width
        for ds in datasets:
            ds.close()
        print(f"[INFO] Done: {os.path.basename(merged_path)}")
