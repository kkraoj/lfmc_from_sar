# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:46:41 2018

@author: kkrao
"""

import os 
import numpy as np
import pandas as pd
import pytz
import datetime
from tzwhere import tzwhere


pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 200)

os.chdir('D:/Krishna/projects/vwc_from_radar')
subset = 'am'
df = pd.read_pickle('data/df_sar_vwc_%s'%subset) #this is the combined dataframe of all queried SAR locations
df['obs_date_local'] = datetime.datetime.now()
ll = pd.read_csv('data/fuel_moisture/nfmd_queried_latlon.csv',index_col = 0) #queried lot lon values

tzwhere = tzwhere.tzwhere()
for index, row in ll.iterrows():
#    if index == 2062:
#        continue
    if not(df.loc[df.Site == str(index),'obs_date'].notnull().all()):
        continue
    tz_str = tzwhere.tzNameAt(row['Latitude'], row['Longitude']) # find time zone based on lat lon
#    print(tz_str)
    if tz_str:
        tz = pytz.timezone(tz_str) #import time into datetime format
        df.loc[df.Site == str(index),'obs_date_local'] = \
        pd.to_datetime(df.loc[df.Site == str(index),'obs_date']+
                       [tz.utcoffset(dt) for dt in
                        df.loc[df.Site == str(index),'obs_date']]) #shift time
        print('[INFO] Time shift done for site %s'%index)
    else:
        print('[INFO] Time shifting failed for site %s'%index)
        
df.obs_date_local.apply(lambda x: x.hour).hist()
df.to_pickle('data/df_sar_vwc_%s'%subset)
