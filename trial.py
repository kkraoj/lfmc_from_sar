# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 07:13:12 2020

@author: kkrao
"""


import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import matplotlib.pylab as plt
from datetime import datetime

from PIL import Image
from numpy import asarray
# load the image
image = Image.open('D:\Krishna\download.jpg')
# convert image to numpy array
data = asarray(image)
print(type(data))
# summarize shape
print(data.shape)


#xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
#    nrows,ncols = np.shape(array)
#    xres = (xmax-xmin)/float(ncols)
#    yres = (ymax-ymin)/float(nrows)
#geotransform=(xmin,xres,0,ymax,0, -yres)   
geotransform=(45,1e-5,0,45,0, -1e-5) 
# That's (top left x, w-e pixel resolution, rotation (0 if North is up), 
#         top left y, rotation (0 if North is up), n-s pixel resolution)
# I don't know why rotation is in twice???

output_raster = gdal.GetDriverByName('GTiff').Create(r'D:\Krishna\bear.tif',data.shape[1], data.shape[0], 3 ,gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
srs = osr.SpatialReference()                 # Establish its coordinate encoding
srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                             # Anyone know how to specify the 
                                             # IAU2000:49900 Mars encoding?
output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system 
                                                   # to the file
output_raster.GetRasterBand(1).WriteArray(data[:,:,0])   # Writes my array to the raster
output_raster.GetRasterBand(2).WriteArray(data[:,:,1])   # Writes my array to the raster
output_raster.GetRasterBand(3).WriteArray(data[:,:,2])   # Writes my array to the raster

output_raster.FlushCache()
output_raster = None