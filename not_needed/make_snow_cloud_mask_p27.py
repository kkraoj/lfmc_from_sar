# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:16:13 2019

@author: kkrao
"""

import arcpy
from arcpy import env
from arcpy.sa import *
import os
import pandas as pd
import gdal

#%% make mask

## Set environment settings
#env.workspace = r"D:\Krishna\projects\vwc_from_radar\data\map\dynamic_maps"
#
## Set local variables
#inRaster = Raster("mask_2018_07_01.tif")
#inTrueRaster = 1
#inFalseConstant = 0
#
## Check out the ArcGIS Spatial Analyst extension license
#arcpy.CheckOutExtension("Spatial")
#
## Execute Con using a map algebra expression instead of a where clause
#outCon = Con(inRaster == 0, inTrueRaster, inFalseConstant)
#
## Save the outputs 
#outCon.save(r"D:\Krishna\projects\vwc_from_radar\data\map\dynamic_maps\mask_classified_2018_07_01.tif")

#%% save mask as csv

os.chdir('D:/Krishna/projects/vwc_from_radar')
### adding different static features
latlon = pd.read_csv('data/map/map_lat_lon.csv', index_col = 0)
def get_value(filename, mx, my, band = 1):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(band).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    return data[py,px]

latlon['mask'] = get_value(r'D:\Krishna\projects\vwc_from_radar\data\map\dynamic_maps\mask_classified_2018_07_01.tif', \
    latlon.longitude.values, latlon.latitude.values)

latlon.to_csv(r'D:\Krishna\projects\vwc_from_radar\data\map\mask_classified_2018_07_01.csv')
