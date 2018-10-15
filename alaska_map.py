# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:34:03 2018

@author: kkrao
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Polygon


def transform(x,y, center_x, center_y, scale, trans_x, trans_y):
    x = scale*(x-center_x)+ trans_x+center_x
    y = scale*(y-center_y)+ trans_y+center_y
    return x,y

#from dirs import dir_data
dir_data = "D:/Krishna/projects/vwc_from_radar/data/fuel_moisture"
os.chdir(dir_data)

files = os.listdir(dir_data+'/raw/')
Df = pd.DataFrame()
for file in files:
    Df = pd.concat([Df, pd.read_table('raw/'+file)])
Df.drop("Unnamed: 7", axis = 1, inplace = True)
Df["Date"] = pd.to_datetime(Df["Date"])

Df['address'] = Df["Site"] + ', '+ Df["State"]
pd.DataFrame(Df['address'].unique(), columns = ['address']).to_csv('address.csv')

## actual latlon queried from nfmd
latlon = pd.read_csv("nfmd_queried_latlon.csv", index_col = 0)
latlon['observations'] = Df.groupby('Site').GACC.count()
latlon.rename(columns = {"Latitude":"latitude", "Longitude":"longitude"}, inplace = True)
temp = Df.drop_duplicates(subset = 'Site')
temp.index = temp.Site
latlon['State'] = np.NaN
latlon.update(temp)
###############################################################################
enlarge = 3.
latlon.sort_values(by=['observations'], ascending = False, inplace = True)
#cmap = 'YlGnBu'
sns.set_style('ticks')
alpha = 1
fig, ax = plt.subplots(figsize=(1*enlarge,1*enlarge))

m = Basemap(llcrnrlon=-120,llcrnrlat=20,urcrnrlon=-110,urcrnrlat=30,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
m.drawmapboundary(fill_color='lightcyan')
#-----------------------------------------------------------------------
# load the shapefile, use the name 'states'
m.readshapefile('D:/Krishna/projects/vwc_from_radar/cb_2017_us_state_500k', 
                name='states', drawbounds=True)
statenames=[]
for shapedict in m.states_info:
    statename = shapedict['NAME']
    statenames.append(statename)
for nshape,seg in enumerate(m.states):
    if statenames[nshape] == 'Alaska':
    # Alaska is too big. Scale it down to 35% first, then transate it. 
        new_seg = [(0.35*args[0] + 1100000, 0.35*args[1]-1500000) for args in seg]
        seg = new_seg    
    poly = Polygon(seg,facecolor='papayawhip',edgecolor='k', zorder  = 1)
    ax.add_patch(poly)
filters = latlon.State == 'AK'
latlon.loc[filters,'longitude'],  latlon.loc[filters,'latitude']=\
          transform(latlon.loc[filters,'longitude'],  latlon.loc[filters,'latitude'],
                     -149.4937,64.2008, 0.35, 38.7,-36.3)   
plot=m.scatter(latlon.longitude.values, latlon.latitude.values, 
               s=8*latlon.observations.values**0.5,c='olivedrab',edgecolor = 'k',\
                    marker='o',alpha = alpha,latlon = True, zorder = 2)
#-----------------------------------------------------------------------
