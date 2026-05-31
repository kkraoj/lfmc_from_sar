"""
Upload LFMC GeoTIFFs to an existing GEE ImageCollection.

Per image:
  1. Upload the TIF to gs://<bucket>/lfmc/<basename>
  2. Submit an ingestion task to GEE: asset id ends with the date,
     system:time_start derived from the YYYY-MM-DD in the filename.

Usage:
  SHERLOCK=1 python upload_lfmc_to_gee.py [--collection <asset_id>]
"""

import argparse
import glob
import os
import re
import sys
import time

import ee
import google.auth.transport.requests
from google.cloud import storage

sys.path.insert(0, os.path.dirname(__file__))
from dirs import dir_data

DEFAULT_COLLECTION = 'projects/earthengine-legacy/assets/users/kkraoj/lfm-mapper/lfmc_col_25_may_2021'
DEFAULT_BUCKET = 'lfmc-inputs'
LFMC_DIR = os.path.join(dir_data, 'map/dynamic_maps/lfmc')

DATE_RE = re.compile(r'lfmc_map_(\d{4}-\d{2}-\d{2})\.tif$')


def get_clients():
    ee.Initialize(project='project-3af726f4-b7ec-4b39-ae4')
    state = ee.data._get_state()
    state.credentials.expiry = None
    state.credentials.refresh(google.auth.transport.requests.Request())
    storage_client = storage.Client(project='project-3af726f4-b7ec-4b39-ae4', credentials=state.credentials)
    return storage_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', default=DEFAULT_COLLECTION)
    parser.add_argument('--bucket', default=DEFAULT_BUCKET)
    parser.add_argument('--gcs-prefix', default='lfmc')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    storage_client = get_clients()
    bucket = storage_client.bucket(args.bucket)

    tifs = sorted(glob.glob(os.path.join(LFMC_DIR, 'lfmc_map_*.tif')))
    print(f'[INFO] Found {len(tifs)} LFMC TIFs in {LFMC_DIR}')

    pairs = []
    for f in tifs:
        m = DATE_RE.search(os.path.basename(f))
        if not m:
            print(f'  skip (no date in name): {os.path.basename(f)}')
            continue
        pairs.append((m.group(1), f))

    print(f'[INFO] {len(pairs)} files to process')
    print(f'[INFO] Target collection: {args.collection}')
    print(f'[INFO] GCS staging: gs://{args.bucket}/{args.gcs_prefix}/')

    if args.dry_run:
        for date, path in pairs:
            print(f'  would upload {date}: {path}')
        return

    task_ids = []
    for date, path in pairs:
        basename = os.path.basename(path)
        gcs_obj = f'{args.gcs_prefix}/{basename}'
        gcs_uri = f'gs://{args.bucket}/{gcs_obj}'
        size_mb = os.path.getsize(path) / 1e6

        # 1. Upload to GCS (skip if same size already there)
        blob = bucket.blob(gcs_obj)
        if blob.exists() and blob.reload() is None and blob.size == os.path.getsize(path):
            print(f'  [{date}] already in GCS ({size_mb:.1f} MB), skipping upload')
        else:
            print(f'  [{date}] uploading {size_mb:.1f} MB -> {gcs_uri} ...')
            blob.upload_from_filename(path)

        # 2. Submit ingestion task
        asset_id = f'{args.collection}/lfmc_map_{date}'
        task_id = ee.data.newTaskId(1)[0]
        request = {
            'name': asset_id,
            'tilesets': [{'sources': [{'uris': [gcs_uri]}]}],
            'startTime': f'{date}T00:00:00Z',
            'properties': {'date': date},
            'pyramidingPolicy': 'MEAN',
        }
        try:
            ee.data.startIngestion(task_id, request, allow_overwrite=True)
            task_ids.append((task_id, date))
            print(f'  [{date}] ingestion task {task_id} submitted -> {asset_id}')
        except Exception as e:
            print(f'  [{date}] INGESTION FAILED: {e}')

    print(f'\n[INFO] {len(task_ids)} ingestion tasks submitted. Poll with `earthengine task list` or in the GEE Code Editor Tasks panel.')


if __name__ == '__main__':
    main()
