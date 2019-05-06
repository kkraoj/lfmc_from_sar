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
from dirs import dir_data
#from osgeo import gdal

#os.chdir('D:/Krishna/projects/vwc_from_radar')
#def get_value(filename, mx, my):
#    ds = gdal.Open(filename)
#    gt = ds.GetGeoTransform()
#    data = ds.GetRasterBand(1).ReadAsArray()
#    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
#    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
#    return data[py,px]
#

#knn = neighbors.KNeighborsRegressor(n_neighbors=10, weights = "uniform", p = 2)
#def seasonal_anomaly(Df, variable, save = False):
#    """
#    index should be datetime
#    
#    """
#    Df['doy'] = Df.date.dt.dayofyear
#
#    sa = pd.DataFrame()
#    for site in Df.site.unique():
#        df = Df.loc[Df.site == site,:]
#        if len(df)<10:
#            continue
#        new_index = df.groupby(df.date).mean().resample('1d').asfreq().index
#        mean = knn.fit(df['doy'].values.reshape(-1,1), df[variable]).predict(np.arange(1, 367).reshape(-1, 1))
#        mean = pd.Series(mean[new_index.dayofyear-1], index = new_index).rolling(window = '10d').mean()       
#        absolute = knn.fit(df.date.values.reshape(-1,1), df[variable]).predict(new_index.values.reshape(-1, 1))
#        absolute = pd.Series(index = new_index, data = absolute).rolling(window = '10d').mean()       
#        anomaly = absolute - mean
#        anomaly.name = site
#        sa = pd.concat([sa,anomaly], axis = 1)
#    sa.name = variable+'_anomaly'
#    if save:
#        sa.to_pickle(os.path.join(dir_data,sa.name))
#    return sa
### add slope ### py 2.7 script
#latlon = pd.read_csv('data/fuel_moisture/nfmd_queried_latlon_10-16-2018.csv', index_col = 0)
#latlon = latlon[latlon.longitude !=0]
#latlon['slope'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\Elevation\Elevation',\
#                            'usa_slope_project.tif'), \
#    latlon.longitude.values, latlon.latitude.values)
#latlon.head()
#latlon.to_csv('data/static_features.csv')
#### add elevation
#latlon = pd.read_csv('data/static_features.csv', index_col = 0)
#latlon['elevation'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\Elevation\Elevation',\
#                            'usa_dem.tif'), \
#    latlon.longitude.values, latlon.latitude.values)
#latlon.head()
#latlon.to_csv('data/static_features.csv')
#### add canopy height
#latlon = pd.read_csv('data/static_features.csv', index_col = 0)
#latlon['canopy_height'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\canopy_height',\
#                            'canopy_height.tif'), \
#    latlon.longitude.values, latlon.latitude.values)
#latlon.head()
#latlon.to_csv('data/static_features.csv')
### add forest_cover
#latlon = pd.read_csv('data/static_features.csv', index_col = 0)
#latlon['forest_cover'] = get_value(os.path.join('D:\Krishna\Project\data\RS_data\Forest\GLOBCOVER',\
#                            'GLOBCOVER_L4_200901_200912_V2.3.tif'), \
#    latlon.longitude.values, latlon.latitude.values)
#latlon.head()
#latlon.to_csv('data/static_features.csv')
## add soil_cover
#latlon = pd.read_csv('data/static_features.csv', index_col = 0)
#for col in ['silt','sand','clay']:   
#    latlon['%s'%col] =\
#      get_value(os.path.join('D:\Krishna\Project\data\RS_data\soil\NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242\data',\
#      'Unified_NA_Soil_Map_Topsoil_%s_Fraction.tif'%col.capitalize()), \
#        latlon.longitude.values, latlon.latitude.values)
#latlon.head()
#latlon.to_csv('data/static_features.csv')


##### add angle
#df = pd.read_pickle('data/df_all')
#df_sar = pd.read_pickle('data/df_sar')
#df = pd.merge(df, df_sar[['Site','angle','obs_date']],  how='left', \
#         left_on=['Site','obs_date'], right_on = ['Site','obs_date'])
#
##### checking if it worked
#pass_type = 'am'
#df = pd.read_pickle('sar_%s_500m'%pass_type)
#df.loc[df.vh<=-30,'vh'] = np.nan
#df.loc[df.vh>0,'vh'] = np.nan
#df.loc[df.vv<=-20,'vv'] = np.nan
#df.loc[df.vv>0,'vv'] = np.nan
#df_angle = df.copy()
#df_angle["vh"] = df.vh*np.cos(np.deg2rad(40))**2/np.cos(np.deg2rad(df.angle))**2
#df_angle["vv"] = df.vv*np.cos(np.deg2rad(40))**2/np.cos(np.deg2rad(df.angle))**2
#df_angle.to_pickle('sar_%s_500m_angle_corr'%pass_type)

#fig, ax = plt.subplots()
#ax.scatter(df['vv'], df_angle['vv'])
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
##### synthetic features
#os.chdir(dir_data)
#ndvi = pd.read_pickle('ndvi_anomaly')
#for param in ['vv','vh']:
#    for pass_type in ['am','pm']:
#        sar = pd.read_pickle('%s_%s_anomaly'%(param,pass_type))
#        df = sar/ndvi
#        df.to_pickle('%s_%s_anomaly_ndvi_anomaly'%(param,pass_type))

######### anomaly of (sar/ndvi)
#opt = pd.read_pickle('opt_500m')
#opt.rename(columns = {'b2':'blue', 'b3':'green', 'b4':'red', 'b8':'nir', 'b11':'swir'}, inplace = True)
#opt = opt.loc[opt.qa60==0,:]
#opt.index = opt.date
#opt['ndvi'] = (opt.nir - opt.red)/(opt.nir + opt.red)
#opt['ndwi'] = (opt.nir - opt.swir)/(opt.nir + opt.swir)
#opt.to_pickle('opt_500m_cloudless')
##site = '17Rd'
##opt.loc[opt.site==site,'ndvi'].plot()/sar.loc[sar.site==site,param].plot()
##gg.plot()
##pass_type = 'am'
#
#for pass_type in ['am','pm']:
#    sar = pd.read_pickle('sar_%s_500m'%(pass_type))
#    sar.index = sar.date
#    #param = 'vh'
#    for param in ['vv','vh']:
#        Df = pd.DataFrame()
#        for site in sar.site.unique(): 
#            df = sar.loc[sar.site==site,param].groupby(sar.loc[sar.site==site,param].index).mean().resample('1d').asfreq().interpolate(method = 'linear')/opt.loc[opt.site==site,'ndvi'].groupby(opt.loc[opt.site==site,'ndvi'].index).mean().resample('1d').asfreq().interpolate(method = 'linear')
#            df = pd.DataFrame(df, columns = ['%s_%s_ndvi'%(param, pass_type)])
#            df['site']=site
#            Df = Df.append(df)
#        Df['date'] = Df.index
#        Df.loc[Df['%s_%s_ndvi'%(param, pass_type)]>0,'%s_%s_ndvi'%(param, pass_type)] = np.nan
#        Df.loc[Df['%s_%s_ndvi'%(param, pass_type)]<-1e3,'%s_%s_ndvi'%(param, pass_type)] = np.nan
#        Df.dropna(inplace = True)
#        Df_a = seasonal_anomaly(Df, '%s_%s_ndvi'%(param, pass_type), save = True )
#
#df = pd.read_pickle('vv_am_ndvi_anomaly')
#df.head()
#pure_species_sites = [
#                     'Ponderosa Basin Ceanothus, Buckbrush New',
#                     'Ponderosa Basin Ceanothus, Buckbrush Old',
#                     'Ponderosa Basin Manzanita, Whiteleaf New',
#                     'Ponderosa Basin Manzanita, Whiteleaf Old']
#for site in pure_species_sites:
#    df[site].plot()
#    plt.show()
#################make dataframe of DOYs
#os.chdir(os.path.join(dir_data, 'cleaned_anomalies_11-29-2018'))
#
#var = "fm_smoothed"
#df = pd.read_pickle(var)
#doy = df.apply(lambda subset: subset.index.dayofyear)
#doy.to_pickle('doy')
##### saved day of year as a dataframe for input
#################make dataframe of DOYsfor all sites
os.chdir(os.path.join(dir_data, 'timeseries'))

var = "vv_smoothed"
df = pd.read_pickle(var)
doy = df.apply(lambda subset: subset.index.dayofyear)
doy.to_pickle('doy')
#### saved day of year as a dataframe for input


