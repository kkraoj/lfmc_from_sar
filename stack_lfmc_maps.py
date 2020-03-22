# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:23:25 2020

@author: kkrao
"""

from osgeo import gdal
import numpy as np
import os
from dirs import dir_data, dir_codes,dir_figures
os.chdir(os.path.join(dir_data, 'map/dynamic_maps/lfmc'))

file = 'lfmc_map_2016-01-01.tif'
ds = gdal.Open(file)
arr = ds.GetRasterBand(1).ReadAsArray()
[nrows, ncols] = arr.shape
output_raster = gdal.GetDriverByName('GTiff').\
        Create(os.path.join(dir_data, 'map\dynamic_maps\lfmc_stack.tif'),\
               ncols, nrows, 4*12*2 ,gdal.GDT_Int16)  # Open the file
output_raster.SetGeoTransform(ds.GetGeoTransform()) # Specify its coordinates
output_raster.SetProjection(ds.GetProjection())   # Exports the coordinate system 
arr = None
ds = None    

ctr = 1
for file in os.listdir(os.path.join(dir_data, 'map/dynamic_maps/lfmc')):#list already ordered
    ds = gdal.Open(file)
    arr = ds.GetRasterBand(1).ReadAsArray()

    output_raster.GetRasterBand(ctr).WriteArray(arr)   # Writes my array to the raster
    arr = None
     
    ds = None
    ctr+=1

output_raster.FlushCache()
output_raster = None   