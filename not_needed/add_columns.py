# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:49:24 2018

@author: kkrao
"""

import numpy as np
import pandas as pd
import os
os.chdir('D:/Krishna/projects/vwc_from_radar')
##### add columns from latlon to df
df = pd.read_pickle('data/df_all')
latlon = pd.read_csv('data/fuel_moisture/nfmd_spatial.csv', index_col = 0)
cols = []
for col in latlon.columns:
    if col in ['observations', 'State']:
        continue
    df[col]=None
    cols.append(col)
for index, row in latlon.iterrows():
    df.loc[df.Site==index,cols] = row[cols].values
df.to_pickle('data/df_all')
#index = 'Wells'
