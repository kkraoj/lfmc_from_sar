# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 08:12:11 2018

@author: kkrao
"""
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from operator import itemgetter


from dirs import dir_data

os.chdir(dir_data)
df= pd.read_pickle('feature_imp_all_sites')

index_type = {'optical': ['blue_anomaly', 'green_anomaly', 'red_anomaly', \
              'nir_anomaly', 'ndvi_anomaly', 'ndwi_anomaly'],\
              'microwave': ['vv_anomaly', 'vh_anomaly', 'vv_ndvi_anomaly', \
                              'vh_ndvi_anomaly']}
static_features = pd.read_csv('static_features.csv',dtype = {'site':str}, index_col = 0)

static_features.loc[static_features.silt<0,['silt','sand','clay']] = 33.3
df[df.abs()>1e2] = np.nan
df.max()
x = df.loc[index_type['microwave']].max()
y = df.loc[index_type['optical']].max()
static_features = static_features.loc[x.index]

color_by_variable = 'canopy_height'

lc_dict = {20: 'crop',
           30: 'crop',
           70: 'closed needleleaf',
           100:'mixed forest',
           110:'shrub',
           120:'shrub',
           130:'shrub',
           140:'grass'}


for color_by_variable in ['forest_cover']:
    
    fig, ax = plt.subplots(figsize = (4,4))
    z = static_features[color_by_variable]
#    classnames, indices = np.unique([lc_dict[z] \
#                      for z in static_features[color_by_variable].values],\
#                                    return_inverse=True)
    df = pd.DataFrame([x,y,z]).T
    df.columns = ['microwave_max','optical_max','forest_cover']
    df.forest_cover = [lc_dict[x] for x in df.forest_cover]
    df.head()
    sns.scatterplot(x="microwave_max", y="optical_max", hue="forest_cover",\
                      data=df, ax = ax)

    
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.08)
#    fig.colorbar(plot,ax=ax,cax=cax)
#    cax.set_ylabel(color_by_variable)    
#    plt.colorbar(ticks=range(7), format=formatter);
    
    axis_range = [-1,8.5]
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.plot(axis_range, axis_range, c = 'grey', lw = 0.5)
#    ax.set_xlabel('Microwave max')
#    ax.set_ylabel('Optical max')
    plt.show()
