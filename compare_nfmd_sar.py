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


dir_data = "D:/Krishna/projects/vwc_from_radar/data/fuel_moisture"
os.chdir(dir_data)
#####-----------------------------------------------------------------
files = os.listdir(dir_data+'/raw/')
Df = pd.DataFrame()
for file in files:
#    sys.stdout.write('\r'+'Processing data for %s ...'%file)
#    sys.stdout.flush()
    Df = pd.concat([Df, pd.read_table('raw/'+file)], \
                    ignore_index = True, sort = False)
#    print(file, Df.shape)
Df.drop("Unnamed: 7", axis = 1, inplace = True)
Df["Date"] = pd.to_datetime(Df["Date"])

meas = Df.copy()
meas.loc[meas.Site == 'Hungry Hunter 33/42','Site'] = 'Hungry Hunter 33_42'
meas = meas.loc[meas.Date>='2014-04-03',:]
meas.reset_index(inplace = True, drop = True)
meas.rename(columns = {"Date":"meas_date"}, inplace = True)
#####-----------------------------------------------------------------
dir_data = 'D:/Krishna/projects/vwc_from_radar/data/sar/'
Df = pd.DataFrame()
for pol in ['VH','HV']:
    files = os.listdir(dir_data+'/'+pol)
    for file in files:
        sys.stdout.write('\r'+'Processing data for %s ...'%file)
        sys.stdout.flush()
        temp = pd.read_csv(dir_data+'/'+pol+'/'+file)
        temp['Site'] = file[:-8]
        Df = pd.concat([Df, temp], axis = 0, ignore_index = True, sort = False)
Df["date"] = pd.to_datetime(Df["date"])
obs = Df.copy()
obs.rename(columns = {"date":"obs_date"}, inplace = True)
#obs.loc[obs.Site == 'Hungry Hunter 33/42','Site'] = 'Hungry Hunter 33_42'
obs.reset_index(inplace = True, drop = True)
#obs.to_pickle('data/df_sar_D')
####------------------------------------------------------------------

meas['VV'] = np.nan; meas['VH'] = np.nan; meas['HH'] = np.nan; meas['HV'] = np.nan; 
meas['obs_date'] = np.nan; meas['residual'] = np.nan
for site in meas.Site.unique():
    print('[INFO] Finding match for site %s'%site)
    if site in obs.Site.unique():
        obs_sub = obs.loc[obs['Site'] == site,:]
        indices, residual = find_nearest(\
                             obs_sub.obs_date,\
                             meas.loc[meas['Site'] == site,'meas_date'] )
        
        meas.loc[meas['Site'] == site,['VV','VH','obs_date']] = \
            obs_sub.loc[obs_sub.index[indices],['VV','VH','obs_date']].values
        meas.loc[meas['Site'] == site,'residual'] = residual
meas.to_pickle("D:/Krishna/projects/vwc_from_radar/data/df_sar_vwc_A")

#####-----------------------------------------------------------------
#lags = []
#for site in obs.Site.unique():
#    print('[INFO] Processing data for Site %s'%site)
#    for meas_date in meas.loc[meas.Site==site,'Date'].unique():
#        lag = min(abs(obs.loc[obs.Site == site,'date']-meas_date))
#        lags.append(lag.days)
#        
#        
#sns.set(font_scale=2)
#sns.set_style('whitegrid')
#n, bins, patches = plt.hist(lags, 50, density=False, facecolor='g', alpha=0.75)
#plt.xlabel('Observation lag (days)')
#plt.ylabel('Measurements')
##plt.title('Histogram of IQ')
##plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
##plt.axis([40, 160, 0, 0.03])
#plt.grid(True)
#plt.show()