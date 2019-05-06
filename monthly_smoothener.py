# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:44:29 2019

@author: kkrao
"""

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from dirs import dir_data, dir_codes
os.chdir(dir_codes)
from QC_of_sites import clean_fmc
os.chdir(dir_data)



def interpolate(df, var = 'percent', ts_start='2015-01-01', ts_end='2018-12-31', window = '30d', max_gap = '120d'):
    df = df.copy()
    df.dropna(subset = [var], inplace = True)
    x_ = df.groupby(df.index).mean().resample('1d').asfreq().index.values
    y_ = df.groupby(df.index).mean().resample('1d').asfreq()[var].interpolate().rolling(window = window).mean()
    z = pd.Series(y_,x_)
    df.sort_index(inplace = True)
    df['delta'] = df['date'].diff()
    gap_end = df.loc[df.delta>=max_gap].index
    gap_start = df.loc[(df.delta>=max_gap).shift(-1).fillna(False)].index
    for start, end in zip(gap_start, gap_end):
        z.loc[start:end] = np.nan
    z =z.reindex(pd.date_range(start=ts_start, end=ts_end, freq='1M'))
    z = pd.DataFrame(z)
    z.dropna(inplace = True)
    z['site'] = df.site[0]
    z['date'] = z.index
    return  z

#%% FMC smoothing
df = pd.read_pickle('fmc_04-29-2019')
df = clean_fmc(df, quality = 'medium')

alpha = 1
for site in ['Reader Ranch']:
    fig, ax = plt.subplots(figsize = (6, 2))
    df_sub = df.loc[df.site==site]
    df_sub.index = df_sub.date
    df_sub.loc[df_sub.fuel==df_sub.fuel.unique()[0]].plot(y = 'percent', marker = 'o', ax= ax, label = df_sub.fuel.unique()[0],\
                linestyle = '', markeredgecolor = 'grey', alpha = alpha )
    df_sub.loc[df_sub.fuel==df_sub.fuel.unique()[1]].plot(y = 'percent', marker = 'o', ax= ax, label = df_sub.fuel.unique()[1],\
                linestyle = '', markeredgecolor = 'grey', alpha = alpha , color ='darkslateblue')
    
    
    monthly = interpolate(df_sub, 'percent')
    monthly.plot(y = 'percent', marker = 'o', ax = ax, label = 'monthly sampled',\
                 linestyle = '', markeredgecolor = 'grey',alpha = alpha)
    ax.set_xlabel('')
    ax.set_ylabel('FMC (%)')
    ax.set_title(site)
    ax.legend(bbox_to_anchor = [0.5,-0.3], loc = 'upper center', ncol = 3)
    plt.show()
    
#%% optical smoothing    
    
#df = pd.read_pickle('landsat8_500m_cloudless')
#alpha = 1
#for site in df.site.unique()[:10]:
#    fig, ax = plt.subplots(figsize = (6, 2))
#    df_sub = df.loc[df.site==site]
#    df_sub.index = df_sub.date
#    df_sub.plot(y = 'red', marker = 'o', ax= ax, label = 'raw',\
#                linestyle = '', markeredgecolor = 'grey', alpha = alpha)
#    
#    
#    monthly = interpolate(df_sub, 'red')
#    monthly.plot(y = 'red', marker = 'o', ax = ax, label = 'monthly sampled',\
#                 linestyle = '', markeredgecolor = 'grey',alpha = alpha)
#    ax.set_xlabel('')
#    ax.set_ylabel('red')
#    ax.set_title(site)
#    ax.legend()
#    plt.show()
#    

#%% sar smoothing    
#    
#df = pd.read_pickle('sar_ascending_30_apr_2019')
#alpha = 1
#var = 'vh'
#for site in df.site.unique()[:10]:
#    fig, ax = plt.subplots(figsize = (6, 2))
#    df_sub = df.loc[df.site==site]
#    df_sub.index = df_sub.date
#    df_sub.plot(y = var, marker = 'o', ax= ax, label = 'raw',\
#                linestyle = '', markeredgecolor = 'grey', alpha = alpha)
#    
#    
#    monthly = interpolate(df_sub, var)
#    monthly.plot(y =var, marker = 'o', ax = ax, label = 'monthly sampled',\
#                 linestyle = '', markeredgecolor = 'grey',alpha = alpha)
#    ax.set_xlabel('')
#    ax.set_ylabel(var)
#    ax.set_title(site)
#    ax.legend()
#    plt.show()