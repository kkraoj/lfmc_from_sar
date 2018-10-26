# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:49:24 2018

@author: kkrao
"""

import numpy as np
import pandas as pd
import os
#from osgeo import gdal

def get_value(filename, mx, my):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(1).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    return data[py,px]
os.chdir('D:/Krishna/projects/vwc_from_radar')
### add slope ### py 2.7 script
latlon = pd.read_csv('data/fuel_moisture/nfmd_queried_latlon_10-16-2018.csv', index_col = 0)
latlon = latlon[latlon.longitude !=0]
latlon['slope'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\Elevation\Elevation',\
                            'usa_slope_project.tif'), \
    latlon.longitude.values, latlon.latitude.values)
latlon.head()
latlon.to_csv('data/static_features.csv')
### add elevation
latlon = pd.read_csv('data/static_features.csv', index_col = 0)
latlon['elevation'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\Elevation\Elevation',\
                            'usa_dem.tif'), \
    latlon.longitude.values, latlon.latitude.values)
latlon.head()
latlon.to_csv('data/static_features.csv')
### add canopy height
latlon = pd.read_csv('data/static_features.csv', index_col = 0)
latlon['canopy_height'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\canopy_height',\
                            'canopy_height.tif'), \
    latlon.longitude.values, latlon.latitude.values)
latlon.head()
latlon.to_csv('data/static_features.csv')
## add forest_cover
latlon = pd.read_csv('data/static_features.csv', index_col = 0)
latlon['forest_cover'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\Forest\GLOBCOVER',\
                            'GLOBCOVER_L4_200901_200912_V2.3.tif'), \
    latlon.longitude.values, latlon.latitude.values)
latlon.head()
latlon.to_csv('data/static_features.csv')
# add soil_cover
latlon = pd.read_csv('data/static_features.csv', index_col = 0)
for col in ['silt','sand','clay']:   
    latlon['%s'%col] =\
      get_value(os.path.join('D:\Krishna\Project\data\RS_data\soil\NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242\data',\
      'Unified_NA_Soil_Map_Topsoil_%s_Fraction.tif'%col.capitalize()), \
        latlon.longitude.values, latlon.latitude.values)
latlon.head()
latlon.to_csv('data/static_features.csv')


##### add angle
#df = pd.read_pickle('data/df_all')
#df_sar = pd.read_pickle('data/df_sar')
#df = pd.merge(df, df_sar[['Site','angle','obs_date']],  how='left', \
#         left_on=['Site','obs_date'], right_on = ['Site','obs_date'])
#
##### checking if it worked
##df_sar[['Site','angle','obs_date']].head()
##df.loc[df.Site == "17Rd",['Site','angle','obs_date']].head()
#df["VH_corr"] = df.VH*np.cos(np.deg2rad(df.angle))
#df.to_pickle('data/df_all')
###### compile sar to 1 pickle
#pass_type = 'pm'
#folder = "sar/500m_ascending"
#files = os.listdir(folder)
#Df = pd.DataFrame()
#for file in files:
##    sys.stdout.write('\r'+'Processing data for %s ...'%file)
##    sys.stdout.flush()
#    df = pd.read_csv('%s/'%folder+file) 
#    df['site'] = file.strip('_gee.csv')
#    Df = Df.append(df, \
#                    ignore_index = True)
##    print(file, Df.shape)
#Df.columns = Df.columns.str.lower()
#Df["date"] = pd.to_datetime(Df["date"])
#Df.date = Df.date.dt.normalize()
#Df['pass_type'] = pass_type
#Df.to_pickle('D:/Krishna/projects/vwc_from_radar/data/sar_%s_500m'%pass_type)

####### compile opt to 1 pickle
#folder = "sentinel2/500m"
#files = os.listdir(folder)
#Df = pd.DataFrame()
#for file in files:
##    sys.stdout.write('\r'+'Processing data for %s ...'%file)
##    sys.stdout.flush()
#    df = pd.read_csv('%s/'%folder+file) 
#    df['site'] = file.strip('_gee.csv')
#    Df = Df.append(df, \
#                    ignore_index = True)
##    print(file, Df.shape)
#Df.columns = Df.columns.str.lower()
#Df["date"] = pd.to_datetime(Df["date"])
#Df.date = Df.date.dt.normalize()
#Df.to_pickle('D:/Krishna/projects/vwc_from_radar/data/opt_500m')
