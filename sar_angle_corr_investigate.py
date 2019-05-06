# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:05:55 2019

@author: kkrao
"""
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from dirs import dir_data



os.chdir(dir_data)
# plotting ts of am and pm vh backscatter to inspect seasonal cycle
site = 15
for site in [1]:
    Df = pd.DataFrame()
    fig, ax = plt.subplots()
    for pass_type in ['pm']:    
        df = pd.read_pickle("sar_%s_500m"%pass_type)
        df.index = df.date
        df.loc[df.site==df.site.unique()[site],'vh'].plot(ax = ax)
#        ax.set_title(df.site.unique()[site])
#        Df = Df.append(df, ignore_index = True)
    #print(Df.head())
    