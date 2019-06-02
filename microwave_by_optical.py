# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 09:28:53 2019

@author: kkrao
"""

import os 
import pandas as pd
import numpy as np
from dirs import dir_data, dir_codes
import matplotlib.pyplot as plt
os.chdir(dir_data)

microwave_inputs = ['vv','vh']
optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv']

resolution = 'SM'
max_gap = '3M'

def interpolate(df, var = 'percent', ts_start='2015-01-01', ts_end='2019-05-31', \
                resolution = '1M',window = '1M', max_gap = '4M'):
    df = df.copy()
    df.index = df.date
    df = df.resample(resolution).mean()

    if resolution == '1M':
        interp_limit = int(max_gap[:-1])
    elif resolution =='SM':
        interp_limit = 2*int(max_gap[:-1])
    else:
        raise  Exception('[INFO] RESOLUTION not supported')
    df = df.interpolate(limit = interp_limit)    
    df = df.dropna()
    df = df[var]
    df['date'] = df.index

    return  df

df = pd.read_pickle('landsat8_500m_cloudless')
# for var in optical_inputs:
opt = pd.DataFrame()
for site in df.site.unique():

    df_sub = df.loc[df.site==site]  
    feature_sub = interpolate(df_sub, var = optical_inputs, resolution = resolution, max_gap = max_gap)
    feature_sub['site'] = site
    opt = opt.append(feature_sub, ignore_index = True, sort = False)

    # master = pd.merge(master,feature, on=['date','site'], how = 'outer')         
### sar
df = pd.read_pickle('sar_ascending_30_apr_2019')
# for var in microwave_inputs:
micro = pd.DataFrame()
for site in df.site.unique():
    df_sub = df.loc[df.site==site]  
    feature_sub = interpolate(df_sub, var = microwave_inputs, resolution = resolution, max_gap = max_gap)
    feature_sub['site'] = site
    micro = micro.append(feature_sub, ignore_index = True, sort = False)    
dyn = pd.merge(opt,micro, on=['date','site'], how = 'outer')
## micro/opt inputs
for num in microwave_inputs:
    for den in optical_inputs:
        dyn['%s_%s'%(num, den)] = dyn[num]/dyn[den]
dyn['vh_vv'] = dyn['vh']-dyn['vv']

# dyn = dyn[dynamic_inputs+['date','site']]
dyn = dyn.dropna()

for site in dyn.site.unique():
    dyn_sub = dyn.loc[dyn.site==site]
    dyn_sub = dyn_sub.sort_values(by = 'date')
    fig, ax = plt.subplots(figsize = (6,2))
    ax.plot(dyn_sub.date, dyn_sub['red'],'r')
    ax.plot(dyn_sub.date, dyn_sub['vh_red'],'m')
    plt.show()
