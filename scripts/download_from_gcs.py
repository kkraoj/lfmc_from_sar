"""
Downloads GEE-exported input files from GCS to the local inputs directory.

Usage:
    SHERLOCK=1 python download_from_gcs.py --year 2025
    SHERLOCK=1 python download_from_gcs.py --year 2025 --year 2026
    SHERLOCK=1 python download_from_gcs.py --year 2025 --bucket lfmc-inputs
"""

import argparse
import os
import sys

import ee
from google.oauth2 import service_account
from google.cloud import storage

sys.path.insert(0, os.path.dirname(__file__))
from dirs import dir_data

KEY_FILE = '/oak/stanford/groups/konings/projects/rao_2020/code/env/gcp_service_account.json'
SERVICE_ACCOUNT = 'lfmc-103@project-3af726f4-b7ec-4b39-ae4.iam.gserviceaccount.com'
PROJECT = 'project-3af726f4-b7ec-4b39-ae4'

LOCAL_DIR = os.path.join(dir_data, 'map/dynamic_maps/inputs_250m')
os.makedirs(LOCAL_DIR, exist_ok=True)


def gcs_client():
    creds = service_account.Credentials.from_service_account_file(KEY_FILE)
    return storage.Client(project=PROJECT, credentials=creds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, action='append', required=True)
    parser.add_argument('--bucket', default='lfmc-inputs')
    args = parser.parse_args()

    client = gcs_client()
    downloaded_blobs = []
    skipped = 0

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
            downloaded_blobs.append(blob)

    print(f'[INFO] Done: {len(downloaded_blobs)} downloaded, {skipped} skipped.')

    if downloaded_blobs:
        print('[INFO] Deleting downloaded blobs from GCS bucket...')
        for blob in downloaded_blobs:
            blob.delete()
            print(f'  Deleted gs://{args.bucket}/{blob.name}')
        print(f'[INFO] Deleted {len(downloaded_blobs)} blobs.')


if __name__ == '__main__':
    main()
