"""
Merges split Landsat-8 GeoTiff tiles exported from GEE into single files.
GEE splits large exports into chunks named like:
  <base>0000000000-0000000000.tif   (new naming — GCS, prefix without trailing dash)
  <base>-0000000000-0000000000.tif  (old naming — Drive, prefix with trailing dash)
The pair `<row>-<col>` is the tile index. Run after downloading from GCS,
before running inference. Chunks are deleted after a successful merge.
"""

import glob
import os
import re
import sys

import rasterio

sys.path.insert(0, os.path.dirname(__file__))
from dirs import dir_data

INPUTS_DIR = os.path.join(dir_data, 'map/dynamic_maps/inputs_250m')
CHUNK_RE = re.compile(r'^(.+?)(\d{10})-(\d{10})\.tif$')


def main():
    groups = {}
    for f in sorted(glob.glob(os.path.join(INPUTS_DIR, '*.tif'))):
        m = CHUNK_RE.match(os.path.basename(f))
        if not m:
            continue
        base = m.group(1).rstrip('-')
        groups.setdefault(base, []).append(f)

    if not groups:
        print("[INFO] No split tiles found.")
        return

    for base, tiles in groups.items():
        tiles.sort()
        merged_path = os.path.join(INPUTS_DIR, base + '.tif')
        print(f"[INFO] Merging {len(tiles)} tiles -> {os.path.basename(merged_path)}")
        for t in tiles:
            print(f"    {os.path.basename(t)}")

        datasets = [rasterio.open(t) for t in tiles]
        try:
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
        finally:
            for ds in datasets:
                ds.close()

        for t in tiles:
            os.remove(t)
        print(f"[INFO] Done: {os.path.basename(merged_path)}  (chunks removed)")


if __name__ == '__main__':
    main()
