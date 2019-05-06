# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:28:18 2019

@author: kkrao
"""
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from dirs import dir_data


os.chdir(dir_data)


smooth_fmc = pd.read_pickle('cleaned_anomalies_11-29-2018/fm_smoothed')


df = pd.read_pickle('vwc')
df = df.loc[df.date.dt.year>=2015]
df = df.loc[~df.fuel.isin(['1-Hour','10-Hour','100-Hour', '1000-Hour', 'Duff (DC)',\
                           '1-hour','10-hour','100-hour', '1000-hour',\
                           'Moss, Dead (DMC)' ])]


df = df.loc[df.site.isin(smooth_fmc.columns)]

df2 = pd.DataFrame()
for site in df.site.unique():
    df_sub = df.loc[df.site==site]
    ##### select fuel with most readings
    df_sub = df_sub.loc[df_sub.fuel==df_sub.fuel.value_counts().idxmax(),:]
    max_count = df_sub.date.value_counts().iloc[0]
    print(max_count)
    if max_count>1:
        raise ValueError("Count exceeded")
    df2 = df2.append(df_sub)
df = df2.copy()
df = df.pivot(index = 'date', columns = 'site',values = 'percent')
#df.index = pd.to_datetime(df.index)
df = df.resample('1D').asfreq()


corr = df.corrwith(smooth_fmc)
#corr.to_pickle('smoothing/discrete_smooth_corr')
fig, ax = plt.subplots(figsize = (14,6))
corr.plot.bar(ax = ax, color = 'k')
ax.set_xlabel('Sites')
ax.set_ylabel('R(discrete, smooth)')
print(corr.describe())

site = "Red Feather"
fig, ax = plt.subplots(figsize = (4,1.5))
smooth_fmc[site].plot(ax = ax)
ax.scatter(df.index, df[site])