"""
Ingest a local LFMC GeoTIFF into a GEE ImageCollection using personal credentials.

WHY THIS SCRIPT EXISTS
----------------------
The target GEE ImageCollection lives under kkraoj's personal Google account
(users/kkraoj/lfm-mapper/lfmc_col_25_may_2021). The project service account
(lfmc-103@project-3af726f4-b7ec-4b39-ae4.iam.gserviceaccount.com) cannot ingest
into that personal namespace even with Writer access granted — GEE legacy asset
permissions don't extend to service account ingestion across personal/project
boundaries. Personal credentials must be used for the GEE step.

TWO-STEP UPLOAD PROCESS
-----------------------
Step 1 — Stage to GCS using the service account (run from scripts/):
    SHERLOCK=1 python upload_lfmc_to_gee.py --date 2026-06-15
    (GCS upload will succeed; GEE ingestion will fail with permission error — ignore it)

Step 2 — Ingest into GEE using personal credentials (this script):
    python upload_to_gee_personal.py --date 2026-06-15
    (requires prior earthengine authentication as kkraoj — see below)

FIRST-TIME AUTHENTICATION (SSH/Sherlock)
-----------------------------------------
Run once per session (or until credentials expire):
    /oak/stanford/groups/konings/projects/rao_2020/code/env/envs/lfmc/bin/python -c \
        "import ee; ee.Authenticate(auth_mode='notebook')"
Open the printed URL in a browser, log in as kkraoj, paste the code back.
Credentials are saved to ~/.config/earthengine/credentials and do NOT affect
the service account key used by all other scripts.

Usage:
    python upload_to_gee_personal.py --date 2026-06-01
"""

import argparse
import os
import sys

import ee

KEY_FILE = '/oak/stanford/groups/konings/projects/rao_2020/code/env/gcp_service_account.json'
DEFAULT_COLLECTION = 'projects/earthengine-legacy/assets/users/kkraoj/lfm-mapper/lfmc_col_25_may_2021'
DEFAULT_BUCKET = 'lfmc-inputs'
DEFAULT_GCS_PREFIX = 'lfmc'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='Date to ingest (YYYY-MM-DD)')
    parser.add_argument('--collection', default=DEFAULT_COLLECTION)
    parser.add_argument('--bucket', default=DEFAULT_BUCKET)
    parser.add_argument('--gcs-prefix', default=DEFAULT_GCS_PREFIX)
    args = parser.parse_args()

    ee.Initialize()

    basename = f'lfmc_map_{args.date}.tif'
    gcs_uri = f'gs://{args.bucket}/{args.gcs_prefix}/{basename}'
    asset_id = f'{args.collection}/lfmc_map_{args.date}'

    print(f'[INFO] GCS source : {gcs_uri}')
    print(f'[INFO] GEE target : {asset_id}')

    task_id = ee.data.newTaskId(1)[0]
    request = {
        'name': asset_id,
        'tilesets': [{'sources': [{'uris': [gcs_uri]}]}],
        'startTime': f'{args.date}T00:00:00Z',
        'properties': {'date': args.date},
        'pyramidingPolicy': 'MEAN',
    }
    ee.data.startIngestion(task_id, request, allow_overwrite=True)
    print(f'[INFO] Ingestion task submitted: {task_id}')
    print(f'[INFO] Monitor in GEE Code Editor Tasks panel.')


if __name__ == '__main__':
    main()
