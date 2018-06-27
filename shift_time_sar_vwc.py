# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:46:41 2018

@author: kkrao
"""

import os 
import glob
import numpy as np
import pandas as pd
import pytz
import datetime
from tzwhere import tzwhere
import operator


pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 200)

os.chdir('D:/Krishna/projects/vwc_from_radar')
df = pd.read_pickle('data/df_sar_vwc_A')
#df = pd.read_pickle('data/df_sar')
df['obs_date_local'] = datetime.datetime.now()
ll = pd.read_csv('data/fuel_moisture/nfmd_queried_latlon.csv',index_col = 0)

tzwhere = tzwhere.tzwhere()
for index, row in ll.iterrows():
#    if index == 2062:
#        continue
    if not(df.loc[df.Site == str(index),'obs_date'].notnull().all()):
        continue
    tz_str = tzwhere.tzNameAt(row['Latitude'], row['Longitude']) # Seville coordinates
#    print(tz_str)
    if tz_str:
        tz = pytz.timezone(tz_str)
        df.loc[df.Site == str(index),'obs_date_local'] = \
        pd.to_datetime(df.loc[df.Site == str(index),'obs_date']+
                       [tz.utcoffset(dt) for dt in
                        df.loc[df.Site == str(index),'obs_date']])
        print('[INFO] Time shift done for site %s'%index)
    else:
        print('[INFO] Time shifting failed for site %s'%index)
        
df.obs_date_local.apply(lambda x: x.hour).hist()
df.to_pickle('data/df_sar_vwc_D')
