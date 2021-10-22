# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:49:24 2018

@author: kkrao
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import neighbors
from osgeo import gdal

os.chdir('D:/Krishna/projects/vwc_from_radar') # root directory

def get_value(filename, mx, my):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(1).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    return data[py,px]

# ## add slope ### requires gdal
# latlon = pd.read_csv('codes/input_data_creation/nfmd_queried_trial.csv', index_col = 0)
# latlon.columns = latlon.columns.str.lower()
# latlon = latlon[latlon.longitude !=0]
# latlon['slope'] = get_value(os.path.join(r'D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\Elevation\Elevation',\
#                             'usa_slope_project.tif'), \
#     latlon.longitude.values, latlon.latitude.values)
# ### add elevation
# latlon['elevation'] = get_value(os.path.join(r'D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\Elevation\Elevation',\
#                             'usa_dem.tif'), \
#     latlon.longitude.values, latlon.latitude.values)
# ### add canopy height
# latlon['canopy_height'] = get_value(os.path.join(r'D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\canopy_height',\
#                             'canopy_height.tif'), \
#     latlon.longitude.values, latlon.latitude.values)
# ## add forest_cover
# latlon['forest_cover'] = get_value(os.path.join(r'D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\Forest\GLOBCOVER',\
#                             'GLOBCOVER_L4_200901_200912_V2.3.tif'), \
#     latlon.longitude.values, latlon.latitude.values)
# # add soil_cover
# for col in ['silt','sand','clay']:   
#     latlon['%s'%col] =\
#       get_value(os.path.join(r'D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\soil\NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242\data',\
#       'Unified_NA_Soil_Map_Topsoil_%s_Fraction.tif'%col.capitalize()), \
#         latlon.longitude.values, latlon.latitude.values)
# latlon.head()
# latlon.to_csv('codes/input_data_creation/static_features.csv')

##### compile sar to 1 pickle
RESOLUTION = 500
os.chdir('D:/Krishna/projects/vwc_from_radar/codes/input_data_creation') # root directory
pass_type = 'pm'
folder = f"S1_{RESOLUTION}m"
files = os.listdir(folder)
Df = pd.DataFrame()
for file in files:
#    sys.stdout.write('\r'+'Processing data for %s ...'%file)
#    sys.stdout.flush()
    df = pd.read_csv('%s/'%folder+file) 
    df['site'] = file.split('_gee.csv')[0]
    Df = Df.append(df, \
                    ignore_index = True)
#    print(file, Df.shape)
Df.columns = Df.columns.str.lower()
Df["date"] = pd.to_datetime(Df["date"])
Df.date = Df.date.dt.normalize()
Df['pass_type'] = pass_type
Df.to_pickle(f'sar_{RESOLUTION}m.pickle')

###### compile opt to 1 pickle
folder = f"L8_{RESOLUTION}m"
files = os.listdir(folder)
Df = pd.DataFrame()
for file in files:
#    sys.stdout.write('\r'+'Processing data for %s ...'%file)
#    sys.stdout.flush()
    df = pd.read_csv('%s/'%folder+file) 
    df['site'] = file.split('_gee.csv')[0]
    Df = Df.append(df, \
                    ignore_index = True)
#    print(file, Df.shape)
Df.columns = Df.columns.str.lower()
Df["date"] = pd.to_datetime(Df["date"])
Df.date = Df.date.dt.normalize()

Df.rename(columns = {'b2':'blue', 'b3':'green', 'b4':'red', 'b5':'nir', 'b6':'swir'}, inplace = True)
cloud_shadow = [328, 392, 840, 904, 1350]
cloud = [352, 368, 416, 432, 480, 864, 880, 928, 944, 992]
high_confidence_cloud = [480, 992] 

all_masked_values = cloud_shadow + cloud + high_confidence_cloud
Df.drop(Df[Df.pixel_qa.isin(all_masked_values)].index, inplace = True) ## in the trial nfmd_query_trial.csv file there are 2 sites. But Vernon site will be dropped because of this line here. There are no cloudless optical data at Vernon Site in Jan 2020

# Df.index = Df.date
Df['ndvi'] = (Df.nir - Df.red)/(Df.nir + Df.red)
Df['ndwi'] = (Df.nir - Df.swir)/(Df.nir + Df.swir)
Df['nirv'] = Df.ndvi*Df.nir
Df.to_pickle(f'opt_{RESOLUTION}m.pickle')
