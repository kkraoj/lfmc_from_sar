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

for pass_type in ['ascending','descending']:
    folder = "sar/500m_%s"%pass_type
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
    Df.to_pickle('sar_%s_30_apr_2019'%pass_type)

###%% ######### check Dfs
for pass_type in ['ascending','descending']:
    Df = pd.read_pickle('sar_%s_30_apr_2019'%pass_type)
    Df.index = Df.date
    fig, ax = plt.subplots(figsize = (9,3))
    Df.loc[Df.site==Df.site.unique()[5],'vh'].plot(marker = 'o',ax = ax,  title = pass_type+'/%s'%Df.site.unique()[5])
    ax.set_xlabel('')
    ax.set_ylabel('vh')
