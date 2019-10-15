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
from matplotlib.collections import PatchCollection
from matplotlib import ticker
from matplotlib.colors import ListedColormap

import seaborn as sns

sns.set_style('ticks')


from keras.models import load_model



os.chdir(dir_data)

pkl_file = open('encoder.pkl', 'rb')
encoder = pickle.load(pkl_file) 
pkl_file.close()

pkl_file = open('scaler.pkl', 'rb')
scaler = pickle.load(pkl_file) 
pkl_file.close()
date = "08-01-2018"
fname = 'map/fmc_map_%s'%date
fname = 'map/fmc_map_2018_07_01_v2'
#%% prediction
# static = pd.read_csv('map/static_features.csv', index_col = 0)
# static.to_pickle('map/static_features')


# static = pd.read_pickle('map/static_features')
# dyn = pd.read_csv('map/dynamic_features_%s.csv'%date, index_col = 0)
# dyn.to_pickle('map/dynamic_features_%s'%date)
# inputs = static.join(dyn.drop(['latitude','longitude'], axis = 1))
# inputs.to_pickle('map/inputs_%s'%date)
# static = None
# dyn = None
# inputs = None

# dataset = pd.read_pickle('map/inputs_%s'%date)
# dataset.drop(['latitude', 'longitude'], axis = 1, inplace = True)
# dataset = dataset.reindex(sorted(dataset.columns), axis=1)

# dataset['percent(t)'] = 100 #dummy
# cols = list(dataset.columns.values)
# cols.remove('percent(t)')
# cols = ['percent(t)']+cols
# dataset = dataset[cols]
# #predictions only on previously trained landcovers
# dataset = dataset.loc[dataset['forest_cover(t)'].astype(int).isin(encoder.classes_)] 
# dataset['forest_cover(t)'] = encoder.transform(dataset['forest_cover(t)'].values)

# for col in dataset.columns:
#     if 'forest_cover' in col:
#         dataset[col] = dataset['forest_cover(t)']
# ##scale
# dataset.replace([np.inf, -np.inf], [1e5, -1e5],inplace = True)
# dataset.fillna(method = 'ffill',inplace = True)
# dataset.fillna(method = 'bfill',inplace = True)


# scaled = scaler.transform(dataset.values)
# dataset.loc[:,:] = scaled
# dataset.drop('percent(t)',axis = 1, inplace = True)
# scaled = dataset.values.reshape((dataset.shape[0], 4, 28), order = 'A')
# np.save('map/scaled_%s.npy'%date, scaled)
# SAVENAME = 'quality_pure+all_same_28_may_2019_res_%s_gap_%s_site_split_raw_ratios'%('1M','3M')
# filepath = os.path.join(dir_codes, 'model_checkpoint/LSTM/%s.hdf5'%SAVENAME)

# model = load_model(filepath)
# yhat = model.predict(scaled)

# scaled = None

# inv_yhat = yhat/scaler.scale_[0]+scaler.min_[0]
# np.save('map/inv_yhat_%s.npy'%date, inv_yhat)
# yhat = None

# dataset = pd.read_pickle('map/inputs_%s'%date)
# #predictions only on previously trained landcovers
# dataset = dataset.loc[dataset['forest_cover(t)'].astype(int).isin(encoder.classes_)] 

# dataset['pred_fmc'] = inv_yhat
# dataset[['latitude','longitude','pred_fmc']].to_pickle(fname)
# dataset = None
# inv_yhat = None

#%% fmc map

latlon = pd.read_pickle(fname)
mask = pd.read_csv('map/mask_classified_2018_07_01.csv')
mask = mask[mask['mask']>0]

enlarge = 1

fig, ax = plt.subplots(figsize=(8*enlarge,7*enlarge))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# load the shapefile, use the name 'states'
m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                    name='states', drawbounds=True)
# m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapfile/states', 
                # name='states', drawbounds=True) 

patches   = []

for info, shape in zip(m.states_info, m.states):
    patches.append(Polygon(np.array(shape), True) )
ax.add_collection(PatchCollection(patches, facecolor= 'grey', edgecolor='k', linewidths=1.5))

colors = ['#703103','#945629','#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', \
          '#74a901', '#66a000', '#529400', '#3e8601', '#207401', '#056201',\
          '#004c00', '#023b01', '#012e01', '#011d01', '#011301']
          
cmap =  ListedColormap(sns.color_palette(colors).as_hex()) 

plot=m.scatter(latlon.longitude.values, latlon.latitude.values, 
              s=0.01,c=latlon.pred_fmc.values,cmap =cmap ,edgecolor = 'w',linewidth = 0,\
                    marker='s',latlon = True, zorder = 2,\
                    vmin = 50, vmax = 200)

#### add mask
m.scatter(mask.longitude.values, mask.latitude.values, 
              s=.1,c='grey',linewidth = 0,edgecolor = 'w',\
                    marker='s',latlon = True, zorder = np.inf)
m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                    name='states', drawbounds=True, linewidth = 1.5)
# m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapfile/states', 
                # name='states', drawbounds=True, linewidth = 1.5) 


cax = fig.add_axes([0.7, 0.5, 0.03, 0.3])
   
cax.annotate('LFMC (%) \n', xy = (0.,1.0), ha = 'left', va = 'bottom')
cb0 = fig.colorbar(plot,ax=ax,cax=cax)
cb0.locator = ticker.MaxNLocator(nbins=5)
cb0.update_ticks()
cb0.set_ticks(np.linspace(50,200,4))
cb0.set_ticklabels(['<50','100','150','>200']) 


tick_labels = [str(int(x)) for x in np.linspace(20,220,6)]
tick_labels[0] = '<50'
tick_labels[-1] = '>200'
cax.set_yticklabels(tick_labels)

# fig.save(r'D:\Krishna\projects\vwc_from_radar\figures\map_high_res.jpg')
#%% . hist of lc 
# lc_dict = {14: 'crop',
#             20: 'crop',
#             30: 'crop',
#             50: 'closed broadleaf deciduous',
#             70: 'closed needleleaf evergreen',
#             90: 'mixed forest',
#             100:'mixed forest',
#             110:'shrub',
#             120:'shrub',
#             130:'shrub',
#             140:'grass',
#             150:'sparse vegetation',
#             160:'regularly flooded forest'}

# dataset = pd.read_pickle('map/inputs')
# dataset = dataset.loc[dataset['forest_cover(t)'].astype(int).isin(encoder.classes_)]
# hist = pd.value_counts(dataset['forest_cover(t)'], normalize = True)
# hist.index = hist.index.to_series().map(lc_dict)
# hist = hist.sort_index()
# fig, ax = plt.subplots(figsize = (4,4))
# hist.plot(kind = 'bar', ax = ax)
# ax.set_ylabel('No. of pixels')


