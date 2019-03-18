# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:25:53 2018

@author: kkrao
"""

import pandas as pd
import matplotlib.pyplot as plt
import os 
from dirs import dir_data
import seaborn as sns

os.chdir(dir_data)
df = pd.read_pickle('vwc')
fuels = ['1-Hour', '10-Hour','1000-Hour', '100-Hour']
df = df.loc[df.fuel.isin(fuels),:]
df.index = df.date
#for site in df.site.unique():
#    df_sub = df.loc[df.site==site,:]
#    
#    if len(df_sub.fuel.unique())<4:
#        continue
#    fig, ax = plt.subplots(figsize = (4,2))
#    ax.set_ylabel('Fuel Moisture (%)')
#    for fuel in df_sub.fuel.unique():
#        ax.plot(df_sub.loc[df_sub.fuel==fuel,'percent'].index, df_sub.loc[df_sub.fuel==fuel,'percent'], label = fuel, marker = 'o', markersize = 4)
#    plt.legend()
#    plt.show()
#
#fig, ax = plt.subplot(figsize = (3,3))    


#df.groupby(['date','fuel']).percent.unique()

df = df.pivot_table(index = ['site','date'] ,columns = 'fuel', values = 'percent')

df.dropna(inplace = True)
df.plot.scatter('1-Hour','1000-Hour')
sns.kdeplot(df['10-Hour'], df['1000-Hour'], n_levels=60, shade=True);