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

sns.set(style = 'ticks',font_scale = 1.5)

#%% clip and mask maps
# county = "marin_sonoma_mendocino_merged_project"
# shapePath = os.path.join(dir_data, "county_shapefiles",county,"%s.shp"%county)
# date = '2019-05-15'
# lfmcPath = os.path.join(dir_data, 'map/dynamic_maps/lfmc/lfmc_map_%s.tif'%date)

# with fiona.open(shapePath, "r") as shapefile:
#     shapes = [feature["geometry"] for feature in shapefile]
    
# with rasterio.open(lfmcPath) as src:
#     out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
#     out_meta = src.meta
    
# out_meta.update({"driver": "GTiff",
#                   "height": out_image.shape[1],
#                   "width": out_image.shape[2],
#                   "transform": out_transform})

# writePath = os.path.join(dir_data,"map","dynamic_maps","lfmc_county","lfmc_%s_%s.tif"%(county,date))
# with rasterio.open(writePath, "w", **out_meta) as dest:
#     dest.write(out_image)
    
#%% plot LFMC maps

dates = ['2019-05-15','2020-05-15','2021-05-15']
county = "marin_sonoma_mendocino_merged_project"
shapePath = os.path.join(dir_data, "county_shapefiles",county,county)

colors = ['#703103','#945629','#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', \
          '#74a901', '#66a000', '#529400', '#3e8601', '#207401', '#056201',\
          '#004c00', '#023b01', '#012e01', '#011d01', '#011301']
cmap =  ListedColormap(colors) 
    
# for date in dates:
#     lfmcPath = os.path.join(dir_data,"map","dynamic_maps","lfmc_county","lfmc_%s_%s.tif"%(county,date))
    
#     ds = gdal.Open(lfmcPath)
#     gt = ds.GetGeoTransform()
#     data = ds.GetRasterBand(1).ReadAsArray().astype(float)
#     data[data<=0] = np.nan 
#     x = np.linspace(start = gt[0],  stop= gt[0]+data.shape[1]*gt[1], num = data.shape[1])    
#     y = np.linspace(start = gt[3],  stop= gt[3]+data.shape[0]*gt[5], num = data.shape[0])    
    
#     x, y = np.meshgrid(x, y)
    
#     fig, ax = plt.subplots(figsize = (6,6))
    
#     m = Basemap(llcrnrlon=-124,llcrnrlat=37.7,urcrnrlon=-122.2,urcrnrlat=40.1,
#             projection='cyl')

          
#     # cmap =  ListedColormap(sns.color_palette("RdYlBu").as_hex()) 
    
#     plot=m.scatter(x,y, zorder = 2, 
#                     s=.1,c=data,cmap =cmap ,linewidth = 0,\
#                         marker='s',latlon = True,\
#                         vmin = 50, vmax = 200)
    
#     #### add mask
    
#     m.readshapefile(shapePath, name='marin_sonoma_mendocino_merged_project', drawbounds=True, linewidth =1)
#     ax.axis('off')
    
#     # cax = fig.add_axes([0.7, 0.4, 0.03, 0.3])
        
#     # # cax.annotate('$\Delta$LFMC (%)', xy = (-1,1.05), ha = 'left', va = 'bottom', color = "k", xycoords = "axes fraction")
#     # cax.annotate('Live fuel\nmoisture (%)', xy = (-1,1.05), ha = 'left', va = 'bottom', color = "k", xycoords = "axes fraction")
#     # cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(50,200,4),extend='both')
#     # # cax.set_yticklabels(['<-20','-10','0','10','>20']) 
#     # cax.set_yticklabels(['<50','100','150','>200']) 
    
    
#     # plt.savefig(os.path.join(dir_figures,'diff_May_2021_2020.jpg'), \
#                                       # dpi =300, bbox_inches="tight")
#     plt.show()



#%% plot LFMC diff maps

cmap =  ListedColormap(sns.color_palette("RdYlBu").as_hex()) 

lfmcPath = os.path.join(dir_data,"map","dynamic_maps","lfmc_county","lfmc_%s_%s.tif"%(county,dates[-1]))
ds = gdal.Open(lfmcPath)
gt = ds.GetGeoTransform()
data3 = ds.GetRasterBand(1).ReadAsArray().astype(float)
data3[data3<=0] = np.nan 
    
for date in dates[:-1]:
    lfmcPath = os.path.join(dir_data,"map","dynamic_maps","lfmc_county","lfmc_%s_%s.tif"%(county,date))
    
    ds = gdal.Open(lfmcPath)
    gt = ds.GetGeoTransform()
    data2 = ds.GetRasterBand(1).ReadAsArray().astype(float)
    data2[data2<=0] = np.nan 
    
    data = data3-data2
    x = np.linspace(start = gt[0],  stop= gt[0]+data.shape[1]*gt[1], num = data.shape[1])    
    y = np.linspace(start = gt[3],  stop= gt[3]+data.shape[0]*gt[5], num = data.shape[0])    
    
    x, y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize = (6,6))
    
    m = Basemap(llcrnrlon=-124,llcrnrlat=37.7,urcrnrlon=-122.2,urcrnrlat=40.1,
            projection='cyl')

    plot=m.scatter(x,y, zorder = 2, 
                    s=.1,c=data,cmap =cmap ,linewidth = 0,\
                        marker='s',latlon = True,\
                        vmin = -20, vmax = 20)
    
    #### add mask
    
    m.readshapefile(shapePath, name='marin_sonoma_mendocino_merged_project', drawbounds=True, linewidth =1)
    ax.axis('off')
    
    cax = fig.add_axes([0.75, 0.4, 0.03, 0.3])
    cax.annotate('Live fuel moisture\ndifference (%)', xy = (0,1.05), ha = 'center', va = 'bottom', color = "k", xycoords = "axes fraction")
    cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(-20,20,5),extend='both')
    cax.set_yticklabels(['<-20','-10','0','10','>20']) 
    
    # plt.savefig(os.path.join(dir_figures,'diff_May_2021_2020.jpg'), \
                                      # dpi =300, bbox_inches="tight")
    plt.show()


