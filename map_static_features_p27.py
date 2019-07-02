# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 05:13:21 2019

@author: kkrao
"""

## add slope ### py 2.7 script
from osgeo import gdal
import os 
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt



os.chdir('D:/Krishna/projects/vwc_from_radar')

### adding different static features
latlon = pd.read_csv('data/map/map_lat_lon.csv', index_col = 0)
def get_value(filename, mx, my):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(1).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    return data[py,px]
#%% add static features
latlon = latlon[latlon.longitude !=0]
latlon['slope'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\Elevation\Elevation',\
                            'usa_slope_project.tif'), \
    latlon.longitude.values, latlon.latitude.values)

### add elevation
latlon['elevation'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\Elevation\Elevation',\
                            'usa_dem.tif'), \
    latlon.longitude.values, latlon.latitude.values)
latlon['canopy_height'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\canopy_height',\
                            'canopy_height.tif'), \
    latlon.longitude.values, latlon.latitude.values)

latlon['forest_cover'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\Forest\GLOBCOVER',\
                            'GLOBCOVER_L4_200901_200912_V2.3.tif'), \
    latlon.longitude.values, latlon.latitude.values)

for col in ['silt','sand','clay']:   
    latlon['%s'%col] =\
      get_value(os.path.join('D:\Krishna\Project\data\RS_data\soil\NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242\data',\
      'Unified_NA_Soil_Map_Topsoil_%s_Fraction.tif'%col.capitalize()), \
        latlon.longitude.values, latlon.latitude.values)

latlon.head()
latlon.to_csv('data/map/static_features.csv')

#%%add dynamic features




#%%
######making map

def plot_map(lats=None,lons=None, var =None,\
             fig = None, ax = None,\
             latcorners = [-90,90], loncorners = [-180, 180],\
             enlarge = 1, marker_factor = 1, \
             cmap = 'YlGnBu', markercolor = 'r',\
             fill = 'papayawhip', background = 'lightcyan',\
             height = 3, width = 5,\
             drawcoast = True, drawcountries = False,\
             drawstates = False, drawcounties = False,\
             shapefilepath = None,shapefilename = None,\
             resolution = 'l', proj = 'cyl', vmin = 0, vmax = 1, **kwargs):
    """
    usage:
    fig, ax = plot_map(lats,lons,var)
    Above inputs are required. 
    
    To add color bar:
        cax = fig.add_axes([0.17, 0.3, 0.03, 0.15])
        fig.colorbar(plot,ax=ax,cax=cax)
    """
    for key, value in kwargs.iteritems():
        print "%s == %s" %(key,value)
    if fig == None:
        fig, ax = plt.subplots(figsize=(width*enlarge,height*enlarge))
    marker_size=get_marker_size(ax,fig,loncorners, marker_factor)
    m = Basemap(projection='lcc',resolution=resolution,\
                    llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=54,
                    lat_1=33,lat_2=45,lon_0=-95,
                    ax=ax, **kwargs)
    if drawcoast:
        m.drawcoastlines()
    if drawcountries:
        m.drawcountries()
    if drawstates:
        m.drawstates()
    if drawcounties:
        m.drawcounties()
    if shapefilepath:
        m.readshapefile(shapefilepath,shapefilename,drawbounds=True, color='black')
    m.fillcontinents(color=fill,zorder=0)
    m.drawmapboundary(fill_color=background, zorder = 0)

    if var is not None:
        plot = m.scatter(lons, lats, s=marker_size,c=var,cmap=cmap,\
                        marker='s', vmin = vmin, vmax = vmax, edgecolor = 'lightgrey',\
                        linewidth = 1)
    else:
        plot = m.scatter(lons, lats, s=marker_size,c=markercolor,\
                        marker='s')
    return fig, ax, m

def get_marker_size(ax,fig,loncorners,grid_size=0.25,marker_factor=1.):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    width *= fig.dpi
    marker_size=width*grid_size/np.diff(loncorners)[0]*marker_factor
    return marker_size


results = pd.read_csv('data/map/results.csv')
results.loc[results.pred_FMC>500,'pred_FMC'] = 50.
fig, ax = plt.subplots(figsize = (8,8))
fig, ax, m = plot_map(results.latitude.values,results.longitude.values, \
                      var =results.clay.values,\
             enlarge = 1, marker_factor = 0.1, \
             cmap = 'magma',\
             fill = 'papayawhip', background = 'lightcyan',\
             height = 1, width = 1,\
             drawcoast = False, drawcountries = True,\
             drawstates = False, drawcounties = False,\
             resolution = 'l', fig = fig, ax = ax, vmin = 0, vmax = 50)
m.scatter(results.longitude.values,results.latitude.values, \
                       s=6,c=results.pred_FMC.values,cmap='magma',\
                        marker='s', vmax = 50, vmin = 0)

#             shapefilepath = r'D:\Krishna\projects\vwc_from_radar\data\usa_shapfile',\
#             shapefilename = "states"
plt.show()

#
#results = pd.read_csv('data/map/results.csv')
#results = results.iloc[:100,:]
#fig, ax = plt.subplots(figsize = (5,5))
#m = Basemap(projection='cyl',
#                llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=54,
#                lat_1=33,lat_2=45,lon_0=-95,
#                ax=ax)
#var =results.clay.values
#m.scatter(results.latitude.values,results.longitude.values, \
#                       s=10000,c=var,cmap='magma',\
#                        marker='s')
#plt.show()
#
#
#results = pd.read_csv('data/map/results.csv')
#var =results.clay.values
#fig, ax = plt.subplots(figsize = (5,5))
#ax.scatter(results.longitude.values,results.latitude.values, \
#                       s=0.3,c=var,cmap='magma',\
#                        marker='s', vmax = 50, vmin = 0)
    