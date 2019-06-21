# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:22:49 2019

@author: kkrao
"""


import os
import sys
from dirs import dir_data
import pandas as pd
import matplotlib.pyplot as plt
import re

os.chdir(dir_data)

passes = ['ascending']
for pass_type in passes:
    folder = "sar/500m_%s_VV"%pass_type
    files = os.listdir(folder)
    Df = pd.DataFrame()
    for file in files:
        sys.stdout.write('\r'+'Processing data for %s ...'%file)
        sys.stdout.flush()
        df = pd.read_csv('%s/'%folder+file) 
    #    df['site'] = file.strip('_gee.csv') ### never use strip
        df['site'] = re.sub(r"\_gee.csv$", "", file)
        Df = Df.append(df, \
                        ignore_index = True)
    #    print(file, Df.shape)
    ## correct date format
    Df["date"] = pd.to_datetime(Df["date"])
    Df.date = Df.date.dt.normalize()
    
    #clean up band names
    Df.columns = Df.columns.str.lower()
    Df.to_pickle('sar_%s_VV_21_jun_2019'%pass_type)

###%% ######### check Dfs
for pass_type in passes:
    Df = pd.read_pickle('sar_%s_VV_21_jun_2019'%pass_type)
    Df.index = Df.date
    fig, ax = plt.subplots(figsize = (9,3))
    Df.loc[Df.site==Df.site.unique()[5],'vv'].plot(marker = 'o',ax = ax,  title = pass_type+'/%s'%Df.site.unique()[5])
    ax.set_xlabel('')
    ax.set_ylabel('vh')

#%%
#### append SAR vv only to previous VH + VV's VV
Df2 = pd.read_pickle('sar_ascending_30_apr_2019')
Df2.drop(['angle','vh'],axis = 1, inplace = True)
Df = pd.read_pickle('sar_ascending_VV_21_jun_2019')  
Df = Df.append(Df2).drop_duplicates()
Df = Df.sort_values(by = ['site','date']) 
Df.to_pickle('sar_ascending_VV_21_jun_2019')
    
