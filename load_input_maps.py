# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 17:06:06 2019

@author: kkrao
"""
import gdal
import arcpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc


inRas = arcpy.Raster(r'D:\Krishna\projects\vwc_from_radar\data\map\dynamic_maps\2018-04-01_sar.tif')
arr = arcpy.RasterToNumPyArray(inRas, nodata_to_value = np.nan)
print(arr.shape)
plt.imshow(arr[1], vmin = -20, vmax = -5)


#inRas = arcpy.Raster(r'D:\Krishna\Project\data\RS_data\Elevation\Elevation\usa_dem.tif')
#
#D:\Krishna\Project\data\RS_data\Forest\GLOBCOVER\
#arr = arcpy.RasterToNumPyArray(inRas, nodata_to_value = -999)
##plt.imshow(arr, vmin = 0, vmax = 4000)
#print(arr.shape)



ds = gdal.Open(r'D:\Krishna\projects\vwc_from_radar\data\map\dynamic_maps\2018-04-01_sar.tif')
geotransform = ds.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]

originX + pixelWidth*6953
originY + pixelHeight*5156

lons = np.linspace(originX,originX + pixelWidth*6953, num = 6953)
lats = np.linspace(originY, originY + pixelHeight*5156, num = 5156)
lons, lats = np.meshgrid(lons,lats)

inRas = arcpy.Raster(r'D:/Krishna/projects/vwc_from_radar/data/usa_shapfile/wu_raster')
roi = arcpy.RasterToNumPyArray(inRas, nodata_to_value = -1)
roi+=1

lons, lats = lons[roi==1], lats[roi==1]
#lons, lats = lons.flatten(), lats.flatten()

latlon = pd.DataFrame(data = {'latitude':lats, 'longitude':lons})
latlon.to_csv(r'D:\Krishna\projects\vwc_from_radar\data\map\map_lat_lon.csv')
gc.collect()