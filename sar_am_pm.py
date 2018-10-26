# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:10:49 2018

@author: kkrao
"""
import os
import pandas as pd 
from dirs import dir_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import neighbors
sns.set(style='ticks')

os.chdir(dir_data)

## plotting ts of am and pm vh backscatter to inspect seasonal cycle
#site = 15
#for site in range(100):
#    Df = pd.DataFrame()
#    for pass_type in ['am','pm']:    
#        df = pd.read_pickle("sar_%s_500m"%pass_type)
#        df = df.loc[df.site==df.site.unique()[site],:]
#        Df = Df.append(df, ignore_index = True)
#    #print(Df.head())
#    
#    #Df.plot('vh')
#    #plt.scatter(Df.date.values, Df.vh.values, color = Df['pass'].values)
#    #plt.plot(Df.vh)
#    #Df = Df.loc[Df.date>='2017-01-01',:]
#    fg = sns.FacetGrid(data=Df, hue='pass' ,aspect=1.61)
#    fg.map(plt.scatter, 'date', 'vh').add_legend()
#    plt.show()
#### am and pm histograms
#fig, ax = plt.subplots()
#for pass_type in ['am','pm']:  
#    df = pd.read_pickle("sar_%s_500m"%pass_type)
#    df.loc[df.vh<=-30,'vh'] = np.nan
#    df.vh.hist(bins = 200, density = True, histtype = 'step', linewidth = 2, ax = ax)
#plt.show()

#### am and pm data availability
#for pass_type in ['am','pm']:  
#    df = pd.read_pickle("sar_%s_500m"%pass_type)
#    print(df.shape)


