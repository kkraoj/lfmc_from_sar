# -*- coding: utf-8 -*-
"""
Exports Sentinel-1 SAR and Landsat-8 images from Google Earth Engine to GCS.

Usage:
    # Export all 1st and 15th of each month for a year:
    python export_gee_maps.py --year 2025

    # Export specific dates:
    python export_gee_maps.py --dates 2025-01-01 2025-02-01 2025-03-01

    # Export and block until all tasks complete:
    python export_gee_maps.py --year 2025 --poll

Each exported date uses a 3-month lookback window for SAR (mosaic) and Landsat
(most-recent-value composite). Output files land in GCS as:
    inputs_250m/YYYY-MM-DD_cloudsnowfree_l8.tif
    inputs_250m/YYYY-MM-DD_sar.tif
"""

import ee
import time
from datetime import date
from ee import batch
from pandas.tseries.offsets import DateOffset
import pandas as pd
import argparse

ee.Initialize(project='project-3af726f4-b7ec-4b39-ae4')

roi = ee.FeatureCollection('users/kkraoj/west_usa')
band = ['VH', 'VV']


def getQABits(image, start, end, newName):
    pattern = 0
    for i in range(start, end + 1):
        pattern += 2 ** i
    return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)

def cloud_shadows(image):
    return getQABits(image.select(['QA_PIXEL']), 4, 4, 'Cloud_shadows').eq(0)

def snow(image):
    return getQABits(image.select(['QA_PIXEL']), 5, 5, 'Snow').eq(0)

def clouds(image):
    return getQABits(image.select(['QA_PIXEL']), 3, 3, 'Cloud').eq(0)

def maskCloudsAndSnow(image):
    image = image.updateMask(cloud_shadows(image))
    image = image.updateMask(snow(image))
    return image.updateMask(clouds(image)).addBands(image.metadata('system:time_start'))


def get_args():
    parser = argparse.ArgumentParser(description='Export SAR + Landsat maps from GEE to GCS')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--year', type=int, help='Export all 1st and 15th dates for this year')
    group.add_argument('--dates', nargs='+', help='Specific end dates to export (YYYY-MM-DD)')
    parser.add_argument('--bucket', default='lfmc-inputs', help='GCS bucket name')
    parser.add_argument('--poll', action='store_true', help='Block until all tasks complete')
    return parser.parse_args()


def poll_tasks(descriptions, interval=30):
    """Poll by description against getTaskList (more reliable than getTaskStatus,
    which sometimes returns stale state for IDs generated client-side)."""
    submitted_at_ms = int(time.time() * 1000) - 10 * 60 * 1000  # 10 min ago, safety margin
    pending = set(descriptions)
    seen = set()
    while pending:
        tasks = ee.data.getTaskList()
        latest = {}
        for t in tasks:
            desc = t.get('description')
            if desc not in pending: continue
            ts = t.get('creation_timestamp_ms', 0)
            if ts < submitted_at_ms: continue
            if desc not in latest or latest[desc].get('creation_timestamp_ms', 0) < ts:
                latest[desc] = t
        for desc, t in latest.items():
            state = t['state']
            if state in ('COMPLETED', 'FAILED', 'CANCELLED') and desc not in seen:
                err = t.get('error_message', '')
                suffix = f" — {err}" if err else ''
                print(f"  Task {desc}: {state}{suffix}")
                pending.discard(desc)
                seen.add(desc)
        if pending:
            print(f"  {len(pending)} tasks still running...")
            time.sleep(interval)
    print('[INFO] All tasks finished.')


def export_dates(end_date_range, bucket, do_poll):
    start_date_range = list((pd.to_datetime(end_date_range) + DateOffset(months=-3)).strftime('%Y-%m-%d'))
    descriptions = []
    for start_date, end_date in zip(start_date_range, end_date_range):
        print(f'[INFO] Queuing export for {end_date} (window: {start_date} to {end_date})')

        image_sar = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
            .select(band).mosaic().clip(roi)

        recentValueComposite = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterDate(start_date, end_date) \
            .filterBounds(roi).map(maskCloudsAndSnow) \
            .qualityMosaic('system:time_start').clip(roi)
        optical = recentValueComposite \
            .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6']) \
            .multiply(0.0000275).add(-0.2) \
            .rename(['B2', 'B3', 'B4', 'B5', 'B6'])

        out = batch.Export.image.toCloudStorage(
            image=optical,
            description=end_date + '_cloudsnowfree_l8',
            bucket=bucket,
            fileNamePrefix='inputs_250m/' + end_date + '_cloudsnowfree_l8',
            scale=250,
            region=roi.geometry().bounds(),
            maxPixels=1e11,
            fileFormat='GeoTIFF',
        )
        batch.Task.start(out)
        descriptions.append(end_date + '_cloudsnowfree_l8')

        out = batch.Export.image.toCloudStorage(
            image=image_sar.select(band),
            description=end_date + '_sar',
            bucket=bucket,
            fileNamePrefix='inputs_250m/' + end_date + '_sar',
            scale=250,
            region=roi.geometry().bounds(),
            maxPixels=1e11,
            fileFormat='GeoTIFF',
        )
        batch.Task.start(out)
        descriptions.append(end_date + '_sar')

    print(f'[INFO] {len(descriptions)} tasks sent to GEE.')
    if do_poll:
        poll_tasks(descriptions)
    else:
        print('[INFO] Monitor progress in the GEE Tasks panel or re-run with --poll.')


if __name__ == '__main__':
    args = get_args()
    if args.year:
        today = date.today().strftime('%Y-%m-%d')
        end_date_range = ['%s-%02d-%02d' % (args.year, month, day)
                          for month in range(1, 13) for day in [1, 15]
                          if '%s-%02d-%02d' % (args.year, month, day) <= today]
    else:
        end_date_range = args.dates
    export_dates(end_date_range, args.bucket, args.poll)
