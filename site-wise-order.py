# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 05:17:41 2018

@author: kkrao
"""

import pandas as pd
import numpy as np
import os 
from dirs import dir_data
import seaborn as sns

os.chdir(dir_data)
df = pd.read_pickle('r2scores')

dynamic_features = ["vwc","sar_am_500m_angle_corr","opt_500m_cloudless"]

static_features = ['slope', 'elevation', 'canopy_height','forest_cover',
                'silt', 'sand', 'clay', 'latitude', 'longitude']

static_features = pd.read_csv('static_features.csv',dtype = {'site':str}, index_col = 0)
df = df.join(static_features)
#df.drop(['Pebble Beach New Growth','Pebble Beach Old Growth' ], inplace = True)
df.drop('train score',axis = 1, inplace = True)
df.sort_values('test score', inplace = True, ascending = False)

for feature in dynamic_features:
    col = pd.read_pickle(feature)
    col = col.loc[col.date>='2015-01-01',:]
    col = col.groupby('site').apply(lambda df: df.shape[0])
    col.name = '%s_no_of_obs'%feature[:3]
    df = df.join(col)
    
    col = pd.read_pickle(feature)
    col = col.loc[col.date>='2015-01-01',:]
    if 'percent' in col.columns:
        col_sub = col.groupby('site').percent.mean()
        col_sub.name = 'fm_mean'
        df = df.join(col_sub)
        col_sub = col.groupby('site').percent.std()
        col_sub.name = 'fm_std'
        df = df.join(col_sub)
    elif 'ndvi' in col.columns:
        for opt_feature in ['ndvi','green','ndwi','nir']:  
            col_sub = col.groupby('site')[opt_feature].mean()
            col_sub.name = '%s_mean'%opt_feature
            df = df.join(col_sub)
            col_sub = col.groupby('site')[opt_feature].std()
            col_sub.name = '%s_std'%opt_feature
            df = df.join(col_sub)
    elif 'vh' in col.columns:
        col_sub = col.groupby('site').vh.mean()
        col_sub.name = 'vh_mean'
        df = df.join(col_sub)              
        col_sub = col.groupby('site').vh.std()
        col_sub.name = 'vh_std'
        df = df.join(col_sub)
df.rename(columns = {'vwc_no_of_obs':'fm_no_of_obs'}, inplace = True)
df.loc[df.sand<0,['sand','silt','clay']] = 33.3
#df.drop('forest_cover',axis = 1, inplace = True)
df.index = range(len(df))
#df.index = range(df.shape[0])
sns.clustermap(df.astype(float), standard_scale =1, row_cluster=False, figsize = (6,4))

