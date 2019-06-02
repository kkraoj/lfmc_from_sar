# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:39:25 2019

@author: kkrao
"""

import os 
import pandas as pd
import numpy as np
from dirs import dir_data, dir_codes
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir(dir_codes)
from QC_of_sites import clean_fmc
os.chdir(dir_data)

df = pd.read_pickle('sar_ascending_30_apr_2019')
df.dropna(inplace = True)


#%% Plot SAR availability per site versus average no. of measurement taken in 2015 (and separately, 2016, 2017, and 2018)
n_site = df.groupby(['site', df.date.dt.year]).vh.count().reset_index().\
    pivot(index = 'site',columns = 'date', values = 'vh')

# fig, ax = plt.subplots(figsize = (3, 6))
# sns.heatmap(n_site, ax =ax)

#%% Plot SAR availability per site versus number of measurements in each site
meas = pd.read_pickle('fmc_24_may_2019')
meas = clean_fmc(meas, quality = 'pure+all same')

n_sar_fmc = pd.DataFrame(meas.groupby('site').percent.count())
n_sar_fmc['sar'] = df.groupby('site').vh.count()
fig, ax = plt.subplots(figsize = (4, 4))
ax.scatter(n_sar_fmc.percent, n_sar_fmc.sar)
ax.set_xlabel('n_fmc')
ax.set_ylabel('n_sar')

#%%Plot SAR availability per site versus duration length of fmc data record

n_sar_fmc['data_record'] = (meas.groupby('site').date.max() - meas.groupby('site').date.min()).dt.days/365
fig, ax = plt.subplots(figsize = (4, 4))
ax.scatter(n_sar_fmc.data_record, n_sar_fmc.sar)
ax.set_xlabel('fmc_data_record (years)')
ax.set_ylabel('n_sar')


n_sar_fmc['fmc_length']
