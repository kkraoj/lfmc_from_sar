# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 04:36:08 2019

@author: kkrao
"""

import pandas as pd
import numpy as np
import os
from dirs import dir_data, dir_codes
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon



os.chdir(dir_data)

# static = pd.read_csv('map/static_features.csv', index_col = 0)
# static.to_pickle('map/static_features')
# dyn = pd.read_csv('map/dynamic_features.csv', index_col = 0)
# dyn.to_pickle('map/dynamic_features')
# inputs = static.join(dyn.drop(['latitude','longitude'], axis = 1))
# inputs.to_pickle('map/inputs')

dataset = pd.read_pickle('map/inputs')
dataset = dataset.reindex(sorted(dataset.columns), axis=1)
dataset.drop(['latitude', 'longitude'], axis = 1, inplace = True)
dataset = dataset.loc[dataset['forest_cover(t)'].astype(int).isin(encoder.classes_)]
dataset['forest_cover(t)'] = encoder.transform(dataset['forest_cover(t)'].values)

for col in dataset.columns:
    if 'forest_cover' in col:
        dataset[col] = dataset['forest_cover(t)']
        
    

##scale

arr = np.array(dataset).reshape((dataset.shape[0], 4, 28), order = 'A')


##### fill all dynamic columns with mean value

#dynamic = pd.read_pickle(r"map\mean_features")
#for feature in dynamic.index:
#    if feature not in static.columns:
#        static[feature] = dynamic[feature]
##        print(feature)
#static.head()
#static.to_csv('map/static_features.csv')
#
######## forward propogate
#static = pd.read_csv('map/static_features.csv')
#feed_data = static.drop('site',axis = 1)
#model = build_model(feed_data.shape[1])
#save_name = 'smoothed_all_sites_8_dec_11_10'
#filepath = os.path.join(dir_codes, 'model_checkpoint/weights_%s.hdf5'%save_name)
#model.load_weights(filepath)
#        # Compile model (required to make predictions)
#model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])    
#pred = model.predict(feed_data).flatten()
#results = static.copy()
#results['pred_FMC'] = pred
#results.to_csv('map/results.csv')



# results = pd.read_csv('map/results.csv')
# enlarge = 1
# cutoff = 2500
# results.loc[results.pred_FMC>cutoff,'pred_FMC'] = cutoff

# fig, ax = plt.subplots(figsize=(8*enlarge,7*enlarge))

# m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=54,
#         projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# m.drawmapboundary(fill_color='lightcyan')
# #-----------------------------------------------------------------------
# # load the shapefile, use the name 'states'
# m.readshapefile('D:/Krishna/projects/vwc_from_radar/cb_2017_us_state_500k', 
#                 name='states', drawbounds=True)
# statenames=[]
# for shapedict in m.states_info:
#     statename = shapedict['NAME']
#     statenames.append(statename)
# for nshape,seg in enumerate(m.states):
#     if statenames[nshape] == 'Alaska':
#     # Alaska is too big. Scale it down to 35% first, then transate it. 
#         new_seg = [(0.35*args[0] + 1100000, 0.35*args[1]-1500000) for args in seg]
#         seg = new_seg    
#     poly = Polygon(seg,facecolor='papayawhip',edgecolor='k', zorder  = 1)
#     ax.add_patch(poly)

# plot=m.scatter(results.longitude.values, results.latitude.values, 
#                s=1,c=results.pred_FMC.values,cmap ='magma' ,edgecolor = 'w',linewidth = 0,\
#                     marker='s',latlon = True, zorder = 2,\
#                     vmin = 0, vmax = cutoff)
# for shapedict in m.states_info:
#     statename = shapedict['NAME']
#     statenames.append(statename)
# for nshape,seg in enumerate(m.states):
#     if statenames[nshape] == 'Alaska':
#     # Alaska is too big. Scale it down to 35% first, then transate it. 
#         new_seg = [(0.35*args[0] + 1100000, 0.35*args[1]-1500000) for args in seg]
#         seg = new_seg    
#     poly = Polygon(seg,facecolor="none",edgecolor='k', zorder  = 3)
#     ax.add_patch(poly)
# #ax.set_title('Length of data record (number of points $\geq$ 10)')
# plt.setp(ax.spines.values(), color='w')
# plt.colorbar()
