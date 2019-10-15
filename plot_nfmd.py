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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Polygon
from pylab import cm
from dirs import dir_figures

# def transform(x,y, center_x, center_y, scale, trans_x, trans_y):
#     x = scale*(x-center_x)+ trans_x+center_x
#     y = scale*(y-center_y)+ trans_y+center_y
#     return x,y
################################################################################
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
################################################################################
#%% plot of nfmd sites with bubbles
#enlarge = 1.
#latlon.sort_values(by=['observations'], ascending = False, inplace = True)
##cmap = 'YlGnBu'
#sns.set_style('ticks')
#alpha = 1
#fig, ax = plt.subplots(figsize=(8*enlarge,5*enlarge))
#
#m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
#m.drawmapboundary(fill_color='lightcyan')
###-----------------------------------------------------------------------
## load the shapefile, use the name 'states'
#m.readshapefile('D:/Krishna/projects/vwc_from_radar/cb_2017_us_state_500k', 
#                name='states', drawbounds=True)
#statenames=[]
#for shapedict in m.states_info:
#    statename = shapedict['NAME']
#    statenames.append(statename)
#for nshape,seg in enumerate(m.states):
#    if statenames[nshape] == 'Alaska':
#    # Alaska is too big. Scale it down to 35% first, then transate it. 
#        new_seg = [(0.35*args[0] + 1100000, 0.35*args[1]-1500000) for args in seg]
#        seg = new_seg    
#    poly = Polygon(seg,facecolor='papayawhip',edgecolor='k', zorder  = 1)
#    ax.add_patch(poly)
#filters = latlon.State == 'AK'
#latlon.loc[filters,'longitude'],  latlon.loc[filters,'latitude']=\
#          transform(latlon.loc[filters,'longitude'],  latlon.loc[filters,'latitude'],
#                     -149.4937,64.2008, 0.35, 38.7,-36.3)   
#plot=m.scatter(latlon.longitude.values, latlon.latitude.values, 
#               s=8*latlon.observations.values**0.5,c='olivedrab',edgecolor = 'k',\
#                    marker='o',alpha = alpha,latlon = True, zorder = 2)
##-----------------------------------------------------------------------
#
#m.scatter(-92.5, 51.2,s=8*1000**0.5,c='olivedrab',edgecolor = 'k',\
#                    marker='o',alpha = alpha,latlon = True) 
#ax.annotate('1000 measurements', xy=(0.6, 0.94), xycoords='axes fraction')  
#plt.setp(ax.spines.values(), color='w')
#plt.show()
#
#latlon.to_csv('nfmd_spatial.csv')
##############################################################################
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
### plot of median delta map
#os.chdir('D:/Krishna/projects/vwc_from_radar')
#df = pd.read_pickle('data/df_sar_vwc_all')
#df.residual = df.residual.abs()
#df = df.loc[df.residual<=2, :]
#
#latlon = pd.read_csv('data/fuel_moisture/nfmd_spatial.csv', index_col = 0)
#latlon['residual'] = df.groupby('Site').residual.mean()
#
#enlarge = 1.
#cmap = cmap = cm.get_cmap('plasma',2)
#sns.set_style('ticks')
#alpha = 1
#fig, ax = plt.subplots(figsize=(8*enlarge,5*enlarge))
#
#m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
#m.drawmapboundary(fill_color='lightcyan')
##-----------------------------------------------------------------------
## load the shapefile, use the name 'states'
#m.readshapefile('D:/Krishna/projects/vwc_from_radar/cb_2017_us_state_500k', 
#                name='states', drawbounds=True)
#statenames=[]
#for shapedict in m.states_info:
#    statename = shapedict['NAME']
#    statenames.append(statename)
#for nshape,seg in enumerate(m.states):
#    if statenames[nshape] == 'Alaska':
#    # Alaska is too big. Scale it down to 35% first, then transate it. 
#        new_seg = [(0.35*args[0] + 1100000, 0.35*args[1]-1500000) for args in seg]
#        seg = new_seg    
#    poly = Polygon(seg,facecolor='papayawhip',edgecolor='k', zorder  = 1)
#    ax.add_patch(poly)
#  
#plot=m.scatter(latlon.longitude.values, latlon.latitude.values, 
#               s=20,c=latlon.residual,cmap =cmap ,edgecolor = 'k',\
#                    marker='o',alpha = alpha,latlon = True, zorder = 2)
#ax.set_title('Median $\Delta$ (days)')
#plt.setp(ax.spines.values(), color='w')
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.08)
#cb=fig.colorbar(plot,ax=ax,cax=cax, ticks = range(3))
##cax.annotate('$\Delta$ days',xy=(0,1.0), xycoords='axes fraction',\
##            ha='left')
#
#plt.show()

##############################################################################
### plot of data record
os.chdir('D:/Krishna/projects/vwc_from_radar')
df = pd.read_pickle('data/df_sar_vwc_all')
df.residual = df.residual.abs()
df = df.loc[df.residual<=2, :]
#df = df.loc[df.data_points>=10, :]
latlon = pd.read_csv('data/fuel_moisture/nfmd_spatial.csv', index_col = 0)
### import 50 sites
selected_sites = pd.read_pickle('data/lstm_input_data_pure+all_same_28_may_2019_res_SM_gap_3M').site.unique()
latlon['color'] = 'lightgrey'
latlon.loc[selected_sites,'color'] = 'maroon'
latlon.sort_values('color', inplace = True)
latlon['data_points'] = df.groupby('Site').obs_date.count()
#latlon = latlon.loc[latlon.data_points>=10,:]
enlarge = 1.7
cmap = 'magma'
sns.set_style('ticks')
alpha = 1
fig, ax = plt.subplots(figsize=(8*enlarge,7*enlarge))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=54,
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
    # if statenames[nshape] == 'Alaska':
    # # Alaska is too big. Scale it down to 35% first, then transate it. 
    #     new_seg = [(0.35*args[0] + 1100000, 0.35*args[1]-1500000) for args in seg]
    #     seg = new_seg    
    poly = Polygon(seg,facecolor='papayawhip',edgecolor='k', zorder  = 1)
    ax.add_patch(poly)
latlon = latlon.loc[selected_sites] # plotting only red sites for IGARSS
plot=m.scatter(latlon.longitude.values, latlon.latitude.values, 
               s=200,c=latlon.color.values,cmap =cmap ,edgecolor = 'w',linewidth = 2,\
                    marker='o',alpha = alpha,latlon = True, zorder = 2,\
                    vmin = 0, vmax = 20)
#ax.set_title('Length of data record (number of points $\geq$ 10)')
plt.setp(ax.spines.values(), color='w')
#plt.legend()
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.08)
#cb=fig.colorbar(plot,ax=ax,cax=cax, ticks = np.linspace(0,20,5))
#cax.annotate('$\Delta$ days',xy=(0,1.0), xycoords='axes fraction',\
#            ha='left')
plt.savefig(os.path.join(dir_figures,'nfmd_sites'), dpi =600,\
                    bbox_inches="tight")
plt.show()