"""
Downloads GEE-exported input files from GCS to the local inputs directory.

Usage:
    SHERLOCK=1 python download_from_gcs.py --year 2025
    SHERLOCK=1 python download_from_gcs.py --year 2025 --year 2026
    SHERLOCK=1 python download_from_gcs.py --year 2025 --bucket ee-kkraoj-inputs
"""

import argparse
import os
import sys

import ee
import google.auth.transport.requests
from google.cloud import storage

sys.path.insert(0, os.path.dirname(__file__))
from dirs import dir_data

LOCAL_DIR = os.path.join(dir_data, 'map/dynamic_maps/inputs_250m')
os.makedirs(LOCAL_DIR, exist_ok=True)


def gcs_client():
    ee.Initialize(project='ee-kkraoj')
    state = ee.data._get_state()
    creds = state.credentials
    creds.expiry = None
    req = google.auth.transport.requests.Request()
    creds.refresh(req)
    return storage.Client(project='ee-kkraoj', credentials=creds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, action='append', required=True)
    parser.add_argument('--bucket', default='ee-kkraoj-inputs')
    args = parser.parse_args()

    client = gcs_client()
    downloaded, skipped = 0, 0

    for year in args.year:
        prefix = f'inputs_250m/{year}-'
        blobs = list(client.list_blobs(args.bucket, prefix=prefix))
        print(f'[INFO] {year}: found {len(blobs)} blobs with prefix {prefix}')
        for blob in blobs:
            local_path = os.path.join(LOCAL_DIR, os.path.basename(blob.name))
            if os.path.exists(local_path) and os.path.getsize(local_path) == blob.size:
                skipped += 1
                continue
            size_gb = (blob.size or 0) / 1e9
            tag = 'Re-downloading (size mismatch)' if os.path.exists(local_path) else 'Downloading'
            print(f'  {tag} {blob.name} ({size_gb:.1f} GB)...')
            blob.download_to_filename(local_path)
            downloaded += 1

    print(f'[INFO] Done: {downloaded} downloaded, {skipped} skipped.')


if __name__ == '__main__':
    main()
