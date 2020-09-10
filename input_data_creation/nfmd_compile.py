# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 04:26:50 2018

@author: kkrao

Script to compile all csv files downloaded using nfmd_download.py into 1 single pandas dataframe
"""

import os
import pandas as pd
from dirs import dir_data

os.chdir(dir_data)
df = pd.DataFrame()
date = '24_may_2019'
for file in os.listdir('fuel_moisture/raw_%s'%date):
    df = df.append([pd.read_table('fuel_moisture/raw_%s/'%date+file)], ignore_index = True)
df.drop('Unnamed: 7', axis = 1, inplace = True)
df.columns = df.columns.str.lower()
df.to_pickle('fmc_%s'%date)
#df.drop_duplicates(subset = 'site').drop(['fuel','percent','date'], axis = 1).to_csv('fuel_moisture/site_info_query_10-16-2018.csv')

