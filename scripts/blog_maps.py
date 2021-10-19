# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:10:43 2021

@author: kkrao
"""

import os
import fiona
import rasterio
import rasterio.mask
from dirs import dir_data, dir_codes,dir_figures
import gdal
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
from plotmap import plotmap


sns.set(style = 'ticks',font_scale = 1.5)

def mask_fires(array):
    maskFilename = "D:/Krishna/projects/vwc_from_radar/blog/california_burned_area_2020.tif"
    lats, lons = get_lats_lons(array)
    mask = get_value(maskFilename, lons, lats, band = 1)
    mask = mask>0
    array[mask]=np.nan
    return array
    

def get_value(filename, mx, my, band = 1):
        ds = gdal.Open(filename)
        gt = ds.GetGeoTransform()
        data = ds.GetRasterBand(band).ReadAsArray()
        px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
        py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
        ds = None
        return data[py,px]
    
def get_lats_lons(array):
    x = range(array.shape[1])
    y = range(array.shape[0])
    
    x,y = np.meshgrid(x,y)
    
    lons = x*gt[1]+gt[0]
    lats = y*gt[5]+gt[3]
    
    return lats, lons

    
#%% plot LFMC maps

dates = ['2021-06-01']
shapePath = "D:/Krishna/projects/vwc_from_radar/data/county_shapefiles/all_counties/CA_counties"
colors = ['#703103','#945629','#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', \
          '#74a901', '#66a000', '#529400', '#3e8601', '#207401', '#056201',\
          '#004c00', '#023b01', '#012e01', '#011d01', '#011301']
cmap =  ListedColormap(colors) 
map_kwargs = dict(llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-114,urcrnrlat=42.5,
        projection='cyl')
scatter_kwargs = dict(cmap = cmap,vmin = 50, vmax =200)


for date in dates:
    lfmcPath = os.path.join(dir_data,"map","dynamic_maps","lfmc_CA","lfmc_CA_%s.tif"%(date))
    
    ds = gdal.Open(lfmcPath)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(1).ReadAsArray().astype(float)
    data[data<=0] = np.nan 
    data = mask_fires(data)
    
    fig, ax = plt.subplots(figsize = (6,6))

    plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = shapePath, 
                  shapefilename ='CA_counties')
    ax.axis("off")
    plt.show()



#%% plot LFMC diff map 2020 - 2021
dates = ['2020-06-01','2021-06-01']
shapePath = "D:/Krishna/projects/vwc_from_radar/data/county_shapefiles/all_counties/CA_counties"

map_kwargs = dict(llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-114,urcrnrlat=42.5,
        projection='cyl')
cmap =  ListedColormap(sns.color_palette("RdYlBu",n_colors = 4).as_hex()) 
scatter_kwargs = dict(cmap = cmap,vmin = -20, vmax =20)

lfmcPath = os.path.join(dir_data,"map","dynamic_maps","lfmc_CA","lfmc_CA_%s.tif"%(dates[-1]))
ds = gdal.Open(lfmcPath)
gt = ds.GetGeoTransform()
data3 = ds.GetRasterBand(1).ReadAsArray().astype(float)
data3[data3<=0] = np.nan 
    
for date in dates[:-1]:
    lfmcPath = os.path.join(dir_data,"map","dynamic_maps","lfmc_CA","lfmc_CA_%s.tif"%(date))
    
    ds = gdal.Open(lfmcPath)
    gt = ds.GetGeoTransform()
    data2 = ds.GetRasterBand(1).ReadAsArray().astype(float)
    data2[data2<=0] = np.nan 
    data = data3-data2
    data = mask_fires(data)
    fig, ax = plt.subplots(figsize = (6,6))

    fig, ax, m, plot = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                  fill = "white",background="white",fig = fig,ax=ax,
                  shapefilepath = shapePath,
              shapefilename ='CA_counties')  
    cax = fig.add_axes([0.65, 0.52, 0.03, 0.3])
    cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(-20,20,5),extend='both')
    cax.set_yticklabels(['<-20','-10','0','10','>20'])
    plt.show()

n= data.shape[0]*data.shape[1]-np.isnan(data).sum()
nLessThanZero = (data<0).sum()/n
print(nLessThanZero)
nLessThanZero = ((data<3)&(data>0)).sum()/n
print(nLessThanZero)
nLessThanZero = (data<-10).sum()/n
print(nLessThanZero)

fig, ax = plt.subplots(figsize = (3,3))
ax.hist(data3.flatten(),color = "m",alpha = 0.5)
ax.hist(data2.flatten(),color = "grey",alpha = 0.5)

#%% 2021 minus all time average

dates = ['2016-06-01','2017-06-01','2018-06-01','2019-06-01','2020-06-01','2021-06-01']
shapePath = "D:/Krishna/projects/vwc_from_radar/data/county_shapefiles/all_counties/CA_counties"

map_kwargs = dict(llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-114,urcrnrlat=42.5,
        projection='cyl')
cmap =  ListedColormap(sns.color_palette("RdYlBu",n_colors = 4).as_hex()) 
scatter_kwargs = dict(cmap = cmap,vmin = -20, vmax =20)

lfmcPath = os.path.join(dir_data,"map","dynamic_maps","lfmc_CA","lfmc_CA_%s.tif"%(dates[-1]))
ds = gdal.Open(lfmcPath)
gt = ds.GetGeoTransform()
data3 = ds.GetRasterBand(1).ReadAsArray().astype(float)
data3[data3<=0] = np.nan 
    
data = np.zeros((len(dates)-1,data3.shape[0],data3.shape[1]))
    
ctr=0
for date in dates[:-1]:
    lfmcPath = os.path.join(dir_data,"map","dynamic_maps","lfmc_CA","lfmc_CA_%s.tif"%(date))
    
    ds = gdal.Open(lfmcPath)
    gt = ds.GetGeoTransform()
    data2 = ds.GetRasterBand(1).ReadAsArray().astype(float)
    data2[data2<=0] = np.nan 
    data[ctr] = data2.copy()
    ctr+=1
    
data = data3-np.nanmean(data,axis=0)
data = mask_fires(data)

fig, ax = plt.subplots(figsize = (6,6))

fig, ax, m, plot = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                  fill = "white",background="white",fig = fig,ax=ax,
                  shapefilepath = shapePath,
              shapefilename ='CA_counties')  
cax = fig.add_axes([0.65, 0.52, 0.03, 0.3])
cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(-20,20,5),extend='both')
cax.set_yticklabels(['<-20','-10','0','10','>20'])
plt.show()

n= data.shape[0]*data.shape[1]-np.isnan(data).sum()
nLessThanZero = (data<0).sum()/n
print(nLessThanZero)
nLessThanZero = ((data<3)&(data>0)).sum()/n
print(nLessThanZero)
nLessThanZero = (data<-10).sum()/n
print(nLessThanZero)

