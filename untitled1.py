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

dynamic_features = ["fm_anomaly","vv_am_anomaly","vh_am_anomaly",\
                    "blue_anomaly","green_anomaly","red_anomaly","nir_anomaly",\
                    'ndvi_anomaly', 'ndwi_anomaly', 'vh_am_ndvi_anomaly', 'vv_am_ndvi_anomaly']

static_features = ['slope', 'elevation', 'canopy_height','forest_cover',
                'silt', 'sand', 'clay', 'latitude', 'longitude']

static_features = pd.read_csv('static_features.csv',dtype = {'site':str}, index_col = 0)
df = df.join(static_features)

df.drop(['Pebble Beach New Growth','Pebble Beach Old Growth' ], inplace = True)
df.drop('train score',axis = 1, inplace = True)
df.sort_values('test score', inplace = True)
sns.clustermap(df.astype(float), standard_scale =1, row_cluster=False)

