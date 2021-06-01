# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 04:36:08 2019

@author: kkrao
"""

import pandas as pd
import numpy as np
import os
import pickle
from dirs import dir_data, dir_codes,dir_figures
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from datetime import datetime 
import gdal

import seaborn as sns

sns.set(style = 'ticks',font_scale = 0.9)


#
enlarge = 1
os.chdir(dir_data)


#%% fmc map
# params = {"ytick.color" : "w",
#           "xtick.color" : "w",
#           "axes.labelcolor" : "w",
#           "axes.edgecolor" : "w"}
# plt.rcParams.update(params)
date = '2021-05-01'
fname = 'map/dynamic_maps/lfmc/lfmc_map_%s.tif'%date

ds = gdal.Open(fname)
gt = ds.GetGeoTransform()
# print(gt)
data = ds.GetRasterBand(1).ReadAsArray().astype(float)
data[data<0] = np.nan

date = '2020-05-01'
fname = 'map/dynamic_maps/lfmc/lfmc_map_%s.tif'%date

ds = gdal.Open(fname)
data2 = ds.GetRasterBand(1).ReadAsArray().astype(float)
data2[data2<0] = np.nan

data = data2 - data[:data2.shape[0],:data2.shape[1]]

    
x = np.linspace(start = gt[0],  stop= gt[0]+data.shape[1]*gt[1], num = data.shape[1])    
y = np.linspace(start = gt[3],  stop= gt[3]+data.shape[0]*gt[5], num = data.shape[0])    

x, y = np.meshgrid(x, y)

x.shape
y.shape
data.shape
# latlon = pd.read_pickle("map/map_lat_lon_p36")

# plt.ioff()
fig, ax = plt.subplots(figsize=(3*enlarge,3*enlarge))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# m = Basemap(llcrnrlon=-123.55,llcrnrlat=38.21,urcrnrlon=-121.77,urcrnrlat=40.45,
#         projection='lcc',lat_1=33,lat_2=45,lon_0=-95) #mendocino fire
# load the shapefile, use the name 'states'
m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                    name='states', drawbounds=True)
# m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapfile/states', 
                # name='states', drawbounds=True) 

patches   = []

for info, shape in zip(m.states_info, m.states):
    patches.append(Polygon(np.array(shape), True) )
ax.add_collection(PatchCollection(patches, facecolor= 'grey', edgecolor='k', linewidths=0.8))

colors = ['#703103','#945629','#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', \
          '#74a901', '#66a000', '#529400', '#3e8601', '#207401', '#056201',\
          '#004c00', '#023b01', '#012e01', '#011d01', '#011301']
          
cmap =  ListedColormap(sns.color_palette("RdYlBu").as_hex()) 

plot=m.scatter(x,y, zorder = 2, 
                s=.1,c=data,cmap ="RdYlBu" ,linewidth = 0,\
                    marker='s',latlon = True,\
                    vmin = -20, vmax = 20)

#### add mask

m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                    name='states', drawbounds=True, linewidth = 0.5)
ax.axis('off')

cax = fig.add_axes([0.7, 0.45, 0.03, 0.3])
    
cax.annotate('$\Delta$LFMC (%)', xy = (-1,1.05), ha = 'left', va = 'bottom', color = "k", xycoords = "axes fraction")
cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(-20,20,5),extend='both')
cax.set_yticklabels(['<-20','-10','0','10','>20']) 


plt.savefig(os.path.join(dir_figures,'diff_May_2021_2020.jpg'), \
                                  dpi =300, bbox_inches="tight")
plt.show()
