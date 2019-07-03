# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 04:36:08 2019

@author: kkrao
"""

import pandas as pd
import numpy as np
import os
import pickle
from dirs import dir_data, dir_codes
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

from keras.models import load_model



os.chdir(dir_data)

pkl_file = open('encoder.pkl', 'rb')
encoder = pickle.load(pkl_file) 
pkl_file.close()

pkl_file = open('scaler.pkl', 'rb')
scaler = pickle.load(pkl_file) 
pkl_file.close()

#%% prediction
# # static = pd.read_csv('map/static_features.csv', index_col = 0)
# # static.to_pickle('map/static_features')
# # dyn = pd.read_csv('map/dynamic_features.csv', index_col = 0)
# # dyn.to_pickle('map/dynamic_features')
# # inputs = static.join(dyn.drop(['latitude','longitude'], axis = 1))
# # inputs.to_pickle('map/inputs')

# dataset = pd.read_pickle('map/inputs')
# dataset.drop(['latitude', 'longitude'], axis = 1, inplace = True)
# dataset = dataset.reindex(sorted(dataset.columns), axis=1)

# dataset['percent(t)'] = 100 #dummy
# cols = list(dataset.columns.values)
# cols.remove('percent(t)')
# cols = ['percent(t)']+cols
# dataset = dataset[cols]

# dataset = dataset.loc[dataset['forest_cover(t)'].astype(int).isin(encoder.classes_)]
# dataset['forest_cover(t)'] = encoder.transform(dataset['forest_cover(t)'].values)

# for col in dataset.columns:
#     if 'forest_cover' in col:
#         dataset[col] = dataset['forest_cover(t)']
# ##scale
# dataset.fillna(method = 'ffill',inplace = True)
# dataset.fillna(method = 'bfill',inplace = True)
# dataset.replace([np.inf, -np.inf], [1e5, -1e5],inplace = True)

# scaled = scaler.transform(dataset.values)
# dataset.loc[:,:] = scaled
# dataset.drop('percent(t)',axis = 1, inplace = True)
# scaled = dataset.values.reshape((dataset.shape[0], 4, 28), order = 'A')
# np.save('map/scaled.npy', scaled)
# SAVENAME = 'quality_pure+all_same_28_may_2019_res_%s_gap_%s_site_split_raw_ratios'%('1M','3M')
# filepath = os.path.join(dir_codes, 'model_checkpoint/LSTM/%s.hdf5'%SAVENAME)

# model = load_model(filepath)
# yhat = model.predict(scaled)

# inv_yhat = yhat/scaler.scale_[0]+scaler.min_[0]
# np.save('map/inv_yhat.npy', inv_yhat)

# latlon = pd.read_csv('map/map_lat_lon.csv', index_col = 0)
# dataset['pred_fmc'] = inv_yhat
# dataset[['latitude','longitude','pred_fmc']].to_pickle('map/fmc_map_2018_07_01')

#%% fmc map

latlon = pd.read_pickle('map/fmc_map_2018_07_01')
enlarge = 1

fig, ax = plt.subplots(figsize=(8*enlarge,7*enlarge))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
m.drawmapboundary(fill_color='white')
# load the shapefile, use the name 'states'
m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapfile/states', 
                name='states', drawbounds=True) 
plot=m.scatter(latlon.longitude.values, latlon.latitude.values, 
              s=0.01,c=latlon.pred_fmc.values,cmap ='magma' ,edgecolor = 'w',linewidth = 0,\
                    marker='s',latlon = True, zorder = 2,\
                    vmin = 40, vmax = 200)
cax = fig.add_axes([0.3, 0.2, 0.03, 0.15])
fig.colorbar(plot,ax=ax,cax=cax)

#%% misc. hist of lc 
lc_dict = {14: 'crop',
            20: 'crop',
            30: 'crop',
            50: 'closed broadleaf deciduous',
            70: 'closed needleleaf evergreen',
            90: 'mixed forest',
            100:'mixed forest',
            110:'shrub',
            120:'shrub',
            130:'shrub',
            140:'grass',
            150:'sparse vegetation',
            160:'regularly flooded forest'}

dataset = pd.read_pickle('map/inputs')
dataset = dataset.loc[dataset['forest_cover(t)'].astype(int).isin(encoder.classes_)]
hist = pd.value_counts(dataset['forest_cover(t)'], normalize = True)
hist.index = hist.index.to_series().map(lc_dict)
hist = hist.sort_index()
fig, ax = plt.subplots(figsize = (4,4))
hist.plot(kind = 'bar', ax = ax)
ax.set_ylabel('No. of pixels')


