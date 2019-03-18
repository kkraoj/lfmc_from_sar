
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:27:49 2018

@author: kkrao
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 200)

def find_nearest(known_array = None, test_array = None):
    known_array = np.array(known_array)
    test_array  = np.array(test_array)
    
    differences = (test_array.reshape(1,-1) - known_array.reshape(-1,1))
    indices = np.abs(differences).argmin(axis=0)
    residual = np.diagonal(differences[indices,])
    residual = residual.astype('timedelta64[D]')/ np.timedelta64(1, 'D')
    residual = residual.astype(int)
    
    return indices, residual


dir_data = "D:/Krishna/projects/vwc_from_radar/data"
os.chdir(dir_data)
#####-----------------------------------------------------------------
files = os.listdir('fuel_moisture/raw')
Df = pd.DataFrame()
for file in files:
#    sys.stdout.write('\r'+'Processing data for %s ...'%file)
#    sys.stdout.flush()
    Df = pd.concat([Df, pd.read_table(os.path.join('fuel_moisture/raw',file))], \
                    ignore_index = True)
#    print(file, Df.shape)
Df.drop("Unnamed: 7", axis = 1, inplace = True)
Df["Date"] = pd.to_datetime(Df["Date"])

meas = Df.copy()
meas.loc[meas.Site == 'Hungry Hunter 33/42','Site'] = 'Hungry Hunter 33_42'
meas = meas.loc[meas.Date>='2014-04-03',:]
meas.reset_index(inplace = True, drop = True)
meas.rename(columns = {"Date":"meas_date"}, inplace = True)
meas.to_pickle('df_vwc')
#####-----------------------------------------------------------------
Df = pd.DataFrame()
files = os.listdir("sentinel2")
for file in files:
    sys.stdout.write('\r'+'Processing data for %s ...'%file)
    sys.stdout.flush()
    temp = pd.read_csv(os.path.join("sentinel2", file))
    temp['Site'] = file[:-8]
    Df = pd.concat([Df, temp], axis = 0, ignore_index = True)
Df["date"] = pd.to_datetime(Df["date"])
obs = Df.copy()
obs.rename(columns = {"date":"obs_date"}, inplace = True)
#obs.loc[obs.Site == 'Hungry Hunter 33/42','Site'] = 'Hungry Hunter 33_42'
obs.reset_index(inplace = True, drop = True)
obs.to_pickle('df_optical')
####------------------------------------------------------------------

bands = ["B2","B3","B4","B8","B11"]
for column in bands+['obs_date', 'residual']:
    meas[column] = np.nan

for site in meas.Site.unique():
    print('[INFO] Finding match for site %s'%site)
    if site in obs.Site.unique():
        obs_sub = obs.loc[obs['Site'] == site,:]
        indices, residual = find_nearest(\
                             obs_sub.obs_date,\
                             meas.loc[meas['Site'] == site,'meas_date'] )
        
        meas.loc[meas['Site'] == site,bands+['obs_date']] = \
            obs_sub.loc[obs_sub.index[indices],bands+['obs_date']].values
        meas.loc[meas['Site'] == site,'residual'] = residual
meas.to_pickle("df_optical_vwc")

#####-----------------------------------------------------------------
os.chdir("D:/Krishna/projects/vwc_from_radar")
df = pd.read_pickle('data/df_optical_vwc')
df['ndvi'] = (df.B8 - df.B4) / (df.B8+df.B4)
df['ndwi'] = (df.B8 - df.B10) / (df.B8+df.B10)
df['nirv'] = df.B8*df.ndvi
df.to_pickle("data/df_optical_vwc")
####---------------------------------------------------------------------------