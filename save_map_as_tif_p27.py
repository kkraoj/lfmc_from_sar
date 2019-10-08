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


#fname = 'map/fmc_map_%s'%date
fname = r'D:\Krishna\projects\vwc_from_radar\data\map\fmc_map_2018_07_01_v2'
latlon = pd.read_csv(fname+'.csv', index_col = 0)
mask = pd.read_csv(r'D:\Krishna\projects\vwc_from_radar\data\map\mask_classified_2018_07_01.csv', index_col = 0)
df = pd.merge(latlon,mask, on=['latitude','longitude'])
df.loc[df['mask']==1,'pred_fmc'] = -9999
df['lat_index'] = df.latitude.rank(method = 'dense', ascending = False).astype(int)-1
df['lon_index'] = df.longitude.rank(method = 'dense', ascending = True).astype(int)-1



#array = np.array(( (0.1, 0.2, 0.3, 0.4),
#                   (0.2, 0.3, 0.4, 0.5),
#                   (0.3, 0.4, 0.5, 0.6),
#                   (0.4, 0.5, 0.6, 0.7),
#                   (0.5, 0.6, 0.7, 0.8) ))
## My image array      
#lat = np.array(( (10.0, 10.0, 10.0, 10.0),
#                 ( 9.5,  9.5,  9.5,  9.5),
#                 ( 9.0,  9.0,  9.0,  9.0),
#                 ( 8.5,  8.5,  8.5,  8.5),
#                 ( 8.0,  8.0,  8.0,  8.0) ))
#lon = np.array(( (20.0, 20.5, 21.0, 21.5),
#                 (20.0, 20.5, 21.0, 21.5),
#                 (20.0, 20.5, 21.0, 21.5),
#                 (20.0, 20.5, 21.0, 21.5),
#                 (20.0, 20.5, 21.0, 21.5) ))

u_lons = np.sort(df.longitude.unique())
u_lats = np.sort(df.latitude.unique())[::-1]
xx, yy = np.meshgrid(u_lons,u_lats)
zz = xx.copy()

zz[:] = -9999
zz[df.lat_index.values,df.lon_index.values] = df.pred_fmc.values
        
    
array = zz.astype(int)
plt.imshow(array,vmin = 40, vmax = 150)
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

output_raster = gdal.GetDriverByName('GTiff').Create(r'D:\Krishna\projects\vwc_from_radar\data\map\myraster.tif',ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
srs = osr.SpatialReference()                 # Establish its coordinate encoding
srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                             # Anyone know how to specify the 
                                             # IAU2000:49900 Mars encoding?
output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system 
                                                   # to the file
output_raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster

output_raster.FlushCache()