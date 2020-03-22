# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:41:35 2020

@author: kkrao
"""

from osgeo import gdal
import numpy as np
import os
from dirs import dir_data, dir_codes,dir_figures
os.chdir(os.path.join(dir_data, 'map/dynamic_maps/lfmc_float'))
for file in os.listdir(os.path.join(dir_data, 'map/dynamic_maps/lfmc_float')):
    ds = gdal.Open(file)
    arr = ds.GetRasterBand(1).ReadAsArray()
    [nrows, ncols] = arr.shape

    output_raster = gdal.GetDriverByName('GTiff').\
        Create(os.path.join(dir_data, 'map\dynamic_maps\lfmc\%s'%file),\
               ncols, nrows, 1 ,gdal.GDT_Int16)  # Open the file
    output_raster.SetGeoTransform(ds.GetGeoTransform()) # Specify its coordinates
    output_raster.SetProjection(ds.GetProjection())   # Exports the coordinate system 
    output_raster.GetRasterBand(1).WriteArray(arr)   # Writes my array to the raster
    arr = None
    output_raster.FlushCache()
    output_raster = None  
    ds = None
    