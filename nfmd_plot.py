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
#Df.index = Df.address
#latlon = pd.read_csv('latlon.csv')
#latlon.drop(0,axis = 0, inplace = True)
#latlon.index = Df['address'].unique()
#latlon.drop(['Unnamed: 0', 'accuracy', 'formatted_address', 'google_place_id',
#       'input_string',  'number_of_results',
#       'postcode', 'status', 'type'],axis = 1, inplace = True)

## actual latlon queried from nfmd
latlon = pd.read_csv("nfmd_queried_latlon.csv", index_col = 0)
latlon['observations'] = Df.groupby('Site').GACC.count()

#latlon.loc[(latlon.latitude<32)& (latlon.longitude < -114),'observations']=np.nan
#latlon.loc[latlon.latitude>55,'longitude'] =latlon.loc[latlon.latitude>55,'longitude']+40
#latlon.loc[latlon.latitude>55,'latitude'] =latlon.loc[latlon.latitude>55,'latitude']-36.
#latlon.head()

###############################################################################

enlarge = 1.
latlon.sort_values(by=['observations'], ascending = False, inplace = True)
#cmap = 'YlGnBu'
sns.set_style('ticks')
#
fig, ax = plt.subplots(figsize=(8*enlarge,5*enlarge))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
m.drawmapboundary(fill_color='aqua')
plot=m.scatter(latlon.longitude.values, latlon.latitude.values, 
               s=8*latlon.observations.values**0.5,c='red',edgecolor = 'k',\
                    marker='o',alpha = 0.8,latlon = True)
# load the shapefile, use the name 'states'
m.readshapefile('D:/Krishna/projects/vwc_from_radar/cb_2017_us_state_500k', 
                name='states', drawbounds=True)
patches   = []

for info, shape in zip(m.states_info, m.states):
    patches.append(Polygon(np.array(shape), True) )  
ax.add_collection(PatchCollection(patches, facecolor= 'papayawhip', edgecolor='k',
                                  linewidths=1., zorder=0))
m.scatter(-86.5, 50.8,s=8*1000**0.5,c='red',edgecolor = 'k',\
                    marker='o',alpha = 0.8,latlon = True) 
ax.annotate('1000 measurements', xy=(0.7, 0.94), xycoords='axes fraction')  
plt.setp(ax.spines.values(), color='w')
#plt.show()

###############################################################################

#fig, ax  = plt.subplots()
#Df.groupby(Df["Date"].dt.year).Date.count().plot(kind="bar")
#ax.set_ylabel('Measurements')

###############################################################################

#filters = (Df.address == 'Yosemite Valley, CA') & (Df.Date.dt.year >= 1997) & (Df.Date.dt.year < 2001) 
#
#plot = Df.loc[filters,:]
#plot.Percent/=100.
#fig, ax  = plt.subplots(figsize = (6,4))
#plot.plot(x='Date',y='Percent', ax = ax, rot = 45, legend = False, style='-o', color = 'k', ms=5)
#ax.set_ylabel('Fuel Moisture')
#
#ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))


##############################################################################

def transform(x,y, center_x, center_y, scale, trans_x, trans_y):
    x = scale*(x-center_x)+ trans_x+center_x
    y = scale*(y-center_y)+ trans_y+center_y
    return x,y
              
statenames=[]
for shapedict in m.states_info:
    statename = shapedict['NAME']
    statenames.append(statename)
for nshape,seg in enumerate(m.states):
    if statenames[nshape] == 'Alaska':
    # Alaska is too big. Scale it down to 35% first, then transate it. 
        seg = list(map(lambda (x,y): (0.35*x + 1100000, 0.35*y-1500000), seg))
    poly = Polygon(seg,facecolor='papayawhip',edgecolor='k',zorder = -1)
    ax.add_patch(poly)

filters = latlon.index.str.endswith('AK')

#
latlon.loc[filters,'longitude'],  latlon.loc[filters,'latitude']=\
          transform(latlon.loc[filters,'longitude'],  latlon.loc[filters,'latitude'],
                     -149.4937,64.2008, 0.35, 38.7,-36.3)
plot=m.scatter(latlon.loc[filters,'longitude'].values, latlon.loc[filters,'latitude'].values, 
               s=8*latlon.loc[filters,'observations'].values**0.5,c='r',edgecolor = 'k',\
                    marker='o',alpha = 0.8,latlon = True)

plt.show()