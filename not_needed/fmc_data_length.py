# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 04:31:56 2019

@author: kkrao
"""

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dirs import dir_data

os.chdir(dir_data)

def longest_non_nans(series):
    a = series.values  # Extract out relevant column from dataframe as array
    m = np.concatenate(( [True], np.isnan(a), [True] ))  # Mask
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1,2)   # Start-stop limits
    start,stop = ss[(ss[:,1] - ss[:,0]).argmax()]  # Get max interval, interval limits
    return stop - start

df = pd.read_pickle('cleaned_anomalies_11-29-2018/fm_smoothed')
length = pd.Series(index = df.columns)


for col in df.columns:
    length.loc[col] = longest_non_nans(df.loc[:,col])/365
    
### bar plot of longest non nan length
    
fig, ax = plt.subplots()
length.index = range(len(length))
length.plot.bar(ax = ax, color= 'k')
ax.set_ylim(0,4)
plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=9)
ax.set_ylabel('Uninterrupted years')
ax.set_xlabel('Pure species sites')

#### dot plot of df where is null
not_null = ~df.isnull()
fig, ax = plt.subplots(figsize = (6,9))
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=9)
sns.heatmap(not_null.T, ax = ax, cmap = 'viridis')
labels = [item.get_text()[:7] for item in ax.get_xticklabels()]
ax.set_xticklabels(labels)

