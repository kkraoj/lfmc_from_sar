# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:31:36 2019

@author: kkrao
"""

import os
import pandas as pd
from dirs import dir_data
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 30)

os.chdir(dir_data)

var = "fm_smoothed"
df = pd.read_pickle('df_vwc_historic')
#df = df.loc[df.meas_date.dt.year>=2015]
df = df.loc[~df.fuel.isin(['1-Hour','10-Hour', '100-Hour', '1000-Hour'])]
df.index = df.meas_date
####plot site vs. no. of species in each site
#df = df.groupby('site').apply(lambda subset: len(subset.fuel.unique()))\
#    .sort_values(ascending = False)
#df.index = range(df.shape[0])
#fig, ax = plt.subplots(figsize = (4,4))
#df.plot(ax = ax)
#ax.set_ylabel('No. of species at site')
#ax.set_xlabel('sites')

######plot mixed site timeseries
sites = df.groupby('site').apply(lambda subset: len(subset.fuel.unique()))\
    .sort_values(ascending = False)
sites = sites.loc[sites>1]
for site in sites.index:
    df_sub = df.loc[df.site==site]
    fig, ax = plt.subplots()
    for fuel in df_sub.fuel.unique():
        df_sub.loc[df_sub.fuel==fuel,'percent'].plot(ax = ax)
    ax.set_ylabel('LFMC (%)')
    ax.set_title(site)
    plt.show()
    
##### plot fmc normalized by mean
#for site in ["Blackberry Hill"]:
#    df_sub = df.loc[df.site==site]
#    fig, ax = plt.subplots()
#    for fuel in df_sub.fuel.unique():
#        df_sub_sub = df_sub.loc[df_sub.fuel==fuel,'percent']
#        df_sub_sub -= df_sub_sub.mean() 
#        df_sub_sub.plot(ax = ax)
#    ax.set_ylabel('LFMC (%)')
#    ax.set_title(site)
#    plt.show()