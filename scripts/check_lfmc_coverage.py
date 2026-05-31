"""
Checks that every month has exactly two LFMC maps (1st and 15th).
Reports months with missing maps.
"""

import os
from collections import defaultdict

LFMC_DIR = '/oak/stanford/groups/konings/projects/rao_2020/data/lfmc_maps'

files = [f for f in os.listdir(LFMC_DIR) if f.startswith('lfmc_map_') and f.endswith('.tif')]

by_month = defaultdict(list)
for f in files:
    date_str = f.replace('lfmc_map_', '').replace('.tif', '')  # YYYY-MM-DD
    ym = date_str[:7]  # YYYY-MM
    day = int(date_str[8:])
    by_month[ym].append(day)

print(f'Total maps: {len(files)}')
print(f'Total months: {len(by_month)}')
print()

issues = []
for ym in sorted(by_month):
    days = sorted(by_month[ym])
    missing = [d for d in [1, 15] if d not in days]
    if missing:
        issues.append((ym, days, missing))

if issues:
    print(f'Months with missing maps ({len(issues)}):')
    for ym, present, missing in issues:
        print(f'  {ym}: has days {present}, missing days {missing}')
else:
    print('All months have maps on both the 1st and 15th.')
