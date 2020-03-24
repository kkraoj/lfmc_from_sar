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
folder = "landsat8/500m"
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
change_names = {'B2':'blue', 'B3':'green', 'B4':'red', 'B5':'nir', 'B6':'swir'}
Df.rename(columns = change_names, inplace = True)
Df.columns = Df.columns.str.lower()
###### synthetic features
Df['ndvi'] = (Df.nir - Df.red)/(Df.nir + Df.red)
Df['ndwi'] = (Df.nir - Df.swir)/(Df.nir + Df.swir)
Df['nirv'] = Df.nir*Df.ndvi
##save work so far
Df.to_pickle('landsat8_500m')
## drop cloudy pixels. source https://www.earthdatascience.org/courses/earth-analytics-python/multispectral-remote-sensing-modis/cloud-masks-with-spectral-data-python/
Df = pd.read_pickle('landsat8_500m')
cloud_shadow = [328, 392, 840, 904, 1350]
cloud = [352, 368, 416, 432, 480, 864, 880, 928, 944, 992]
high_confidence_cloud = [480, 992]

all_masked_values = cloud_shadow + cloud + high_confidence_cloud
Df.drop(Df[Df.pixel_qa.isin(all_masked_values)].index, inplace = True)
Df.shape
Df.to_pickle('landsat8_500m_cloudless')

######## check Dfs
for dataset in ['landsat8_500m_cloudless', 'landsat8_500m']:
    Df = pd.read_pickle(dataset)
    Df.index = Df.date
    fig, ax = plt.subplots(figsize = (9,3))
    Df.loc[Df.site==Df.site.unique()[5],'red'].plot(marker = 'o', ylim = [400,1200], ax = ax,  title = dataset+'/%s'%Df.site.unique()[5])
    ax.set_xlabel('')
    ax.set_ylabel('red')
