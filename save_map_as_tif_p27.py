# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 07:14:21 2019

@author: kkrao
"""


import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import matplotlib.pylab as plt
from datetime import datetime



for MoY in range(1, 13):
    date = '2019-%02d-01'%(MoY)
    print('[INFO] Making lfmc tif for %s at %s'%(date,datetime.now().strftime("%H:%M:%S")))
    #fname = 'map/fmc_map_%s'%date
    fname = r'D:\Krishna\projects\vwc_from_radar\data\map\dynamic_maps\fmc_map_%s'%date
    # latlon = pd.read_csv(fname+'.csv', index_col = 0)
    df = pd.read_pickle(fname)
    # mask = pd.read_csv(r'D:\Krishna\projects\vwc_from_radar\data\map\mask_classified_2018_07_01.csv', index_col = 0)
    # df = pd.merge(latlon,mask, on=['latitude','longitude'])
    # df.loc[df['mask']==1,'pred_fmc'] = -9999
    df['lat_index'] = df.latitude.rank(method = 'dense', ascending = False).astype(int)-1
    df['lon_index'] = df.longitude.rank(method = 'dense', ascending = True).astype(int)-1
    
    
    u_lons = np.sort(df.longitude.unique())
    u_lats = np.sort(df.latitude.unique())[::-1]
    xx, yy = np.meshgrid(u_lons,u_lats)
    zz = xx.copy()
    
    zz[:] = -9999
    zz[df.lat_index.values,df.lon_index.values] = df.pred_fmc.values
            
        
    array = zz.astype(int)
    fig, ax = plt.subplots(figsize = (2,2))
    ax.imshow(array,vmin = 50, vmax = 200)
    ax.set_title(date)
    plt.show()
    lat = yy
    lon = xx
    # For each pixel I know it's latitude and longitude.
    # As you'll see below you only really need the coordinates of
    # one corner, and the resolution of the file.
    
    xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
    nrows,ncols = np.shape(array)
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)   
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up), 
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    # I don't know why rotation is in twice???
    
    output_raster = gdal.GetDriverByName('GTiff').Create(r'D:\Krishna\projects\vwc_from_radar\data\map\dynamic_maps\lfmc_map_%s.tif'%date,ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                 # Anyone know how to specify the 
                                                 # IAU2000:49900 Mars encoding?
    output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system 
                                                       # to the file
    output_raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster
    
    output_raster.FlushCache()
    output_raster = None
