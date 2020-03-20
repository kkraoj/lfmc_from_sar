# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 05:13:21 2019

@author: kkrao
"""

## add slope ### py 2.7 script
from osgeo import gdal, osr
import os 
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pickle
from dirs import dir_data, dir_codes,dir_figures
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from datetime import datetime 
import seaborn as sns

sns.set_style('ticks')


from keras.models import load_model


enlarge = 1
os.chdir(dir_data)

pkl_file = open('encoder.pkl', 'rb')
encoder = pickle.load(pkl_file) 
pkl_file.close()

pkl_file = open('scaler.pkl', 'rb')
scaler = pickle.load(pkl_file) 
pkl_file.close()

SAVENAME = 'quality_pure+all_same_28_may_2019_res_%s_gap_%s_site_split_raw_ratios'%('1M','3M')
filepath = os.path.join(dir_codes, 'model_checkpoint/LSTM/%s.hdf5'%SAVENAME)

model = load_model(filepath)
os.chdir('D:/Krishna/projects/vwc_from_radar')

### adding different static features
latlon = pd.read_csv(os.path.join(dir_data, 'map/map_lat_lon.csv'), index_col = 0)
def get_value(filename, mx, my, band = 1):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(band).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    return data[py,px]
#static = pd.read_csv(os.path.join(dir_data, 'map/static_features.csv'), index_col = 0)
#static.to_pickle(os.path.join(dir_data, 'map/static_features_p36'))
static = pd.read_pickle(os.path.join(dir_data, 'map/static_features_p36'))
#%%add dynamic features
year = 2018
day = 1

for MoY in range(12, 0, -1):
#    latlon = pd.read_csv('data/map/map_lat_lon.csv', index_col = 0)
#    latlon.to_pickle(os.path.join(dir_data, 'map/map_lat_lon_p36'))
    latlon = pd.read_pickle(os.path.join(dir_data, 'map/map_lat_lon_p36'))
    date = '%04d-%02d-%02d'%(year, MoY, day)
    
    if os.path.exists(os.path.join(dir_data, 'map\dynamic_maps\lfmc\lfmc_map_%s.tif'%date)):
        print('[INFO] Skipping %s because LFMC map already exists.'%(date))
        continue
    
    print('[INFO] Making feature file for %s at %s'%(date,datetime.now().strftime("%H:%M:%S")))
    ####sar
    raw_opt_bands = ['blue','green','red','nir','swir']
    raw_sar_bands = ['vh','vv']
    ## check if all lag files exist
    lag_dates = [(pd.to_datetime(date) - pd.DateOffset(months = lag)).strftime('%Y-%m-%d') for lag in range(3, -1, -1)]
    file_exists = [os.path.exists(os.path.join(dir_data, 'map\dynamic_maps\%s_cloudsnowfree_l8.tif'%lag_date)) for lag_date in lag_dates]
    file_exists += [os.path.exists(os.path.join(dir_data, 'map\dynamic_maps\%s_sar.tif'%lag_date)) for lag_date in lag_dates]   
    if all(file_exists)==False:
        print('[INFO] Skipping %s because at least 1 file does not exist'%(date))
        continue
    
    for lag in range(3, -1, -1):
        lag_date = (pd.to_datetime(date) - pd.DateOffset(months = lag)).strftime('%Y-%m-%d')
        band_names = dict(zip(range(1,6),raw_opt_bands))
        for band in band_names.keys():
            latlon[band_names[band]] = get_value(os.path.join(dir_data, 'map\dynamic_maps\%s_cloudsnowfree_l8.tif'%lag_date),\
            latlon.longitude.values, latlon.latitude.values, band = band)
        latlon.update(latlon.filter(raw_opt_bands).clip(lower = 0))
        
        latlon['ndvi'] = (latlon.nir - latlon.red)/(latlon.nir + latlon.red)
        latlon['ndwi'] = (latlon.nir - latlon.swir)/(latlon.nir + latlon.swir)
        latlon['nirv'] = latlon.nir*latlon.ndvi
        
        band_names = dict(zip(range(1,3),raw_sar_bands))
        for band in band_names.keys():
            latlon[band_names[band]] = get_value(os.path.join(dir_data, 'map\dynamic_maps\%s_sar.tif'%lag_date),\
            latlon.longitude.values, latlon.latitude.values, band = band)
        latlon.update(latlon.filter(raw_sar_bands).clip(upper = 0))
        latlon['vh_vv'] = latlon.vh - latlon.vv
        
        ## mixed inputs
        for num in raw_sar_bands:
            for den in raw_opt_bands:
                latlon['%s_%s'%(num, den)] = latlon[num]/latlon[den]
        
        if lag!=0:
            latlon.columns=list(latlon.columns[:-21])+list(latlon.columns[-21:]+'(t-%d)'%lag)
        else:
            latlon.columns=list(latlon.columns[:-21])+list(latlon.columns[-21:]+'(t)')
    
#    latlon.to_csv('data/map/map_features/dynamic_features_%s.csv'%date)
#    rather than saving the input feature file, just do the prediction here itself to save hard disk space
    print('[INFO] Making lfmc map for %s at %s'%(date,datetime.now().strftime("%H:%M:%S")))
    fname = 'map/dynamic_maps/fmc_map_%s'%date
#    dyn = pd.read_csv('map/map_features/dynamic_features_%s.csv'%date, index_col = 0)
    # dyn.to_pickle('map/dynamic_features_%s'%date)
    dataset = static.join(latlon.drop(['latitude','longitude'], axis = 1))
    # inputs.to_pickle('map/inputs_%s'%date)
    # static = None
    latlon = None
    # inputs = None
   
    # dataset = pd.read_pickle('map/inputs_%s'%date)
    # dataset.drop(['latitude', 'longitude'], axis = 1, inplace = True)
    dataset = dataset.reindex(sorted(dataset.columns), axis=1)
   
    ### add percent col to start
    dataset['percent(t)'] = 100 #dummy
    cols = list(dataset.columns.values)
    cols.remove('percent(t)')
    cols.remove('latitude')
    cols.remove('longitude')
    cols = ['latitude', 'longitude','percent(t)']+cols
    dataset = dataset[cols]
   
    #predictions only on previously trained landcovers
    dataset = dataset.loc[dataset['forest_cover(t)'].astype(int).isin(encoder.classes_)] 
    dataset['forest_cover(t)'] = encoder.transform(dataset['forest_cover(t)'].values)
   
    for col in dataset.columns:
        if 'forest_cover' in col:
            dataset[col] = dataset['forest_cover(t)']
   
    ##scale
    dataset.replace([np.inf, -np.inf], [1e5, -1e5],inplace = True)
    dataset.dropna(inplace = True)
    # dataset.fillna(method = 'ffill',inplace = True)
    # dataset.fillna(method = 'bfill',inplace = True)
    scaled = scaler.transform(dataset.drop(['latitude','longitude'],axis = 1).values)
    dataset.loc[:,2:] = scaled #skip latlon
    dataset.drop('percent(t)',axis = 1, inplace = True)
    scaled = dataset.drop(['latitude','longitude'],axis=1).values.reshape((dataset.shape[0], 4, 28), order = 'A') #langs x features
    # np.save('map/scaled_%s.npy'%date, scaled)
   
    yhat = model.predict(scaled)
   
    scaled = None
   
    inv_yhat = yhat/scaler.scale_[0]+scaler.min_[0]
    # np.save('map/inv_yhat_%s.npy'%date, inv_yhat)
    yhat = None
   
    # dataset = pd.read_pickle('map/inputs_%s'%date)
    #predictions only on previously trained landcovers
    # dataset = dataset.loc[dataset['forest_cover(t)'].astype(int).isin(encoder.classes_)] 
   
    dataset['pred_fmc'] = inv_yhat
#    dataset[['latitude','longitude','pred_fmc']].to_pickle(fname)
    df = dataset[['latitude','longitude','pred_fmc']]
    dataset = None
    inv_yhat = None
    print('[INFO] Saving lfmc map for %s at %s'%(date,datetime.now().strftime("%H:%M:%S")))
    df['lat_index'] = df.latitude.rank(method = 'dense', ascending = False).astype(int)-1
    df['lon_index'] = df.longitude.rank(method = 'dense', ascending = True).astype(int)-1
   
  
    u_lons = np.sort(df.longitude.unique())
    u_lats = np.sort(df.latitude.unique())[::-1]
    xx, yy = np.meshgrid(u_lons,u_lats)
    zz = xx.copy()
    u_lons = None
    u_lats = None
    zz[:] = -9999
    zz[df.lat_index.values,df.lon_index.values] = df.pred_fmc.values
           
    df = None
    array = zz.astype(int)
    zz = None
#    fig, ax = plt.subplots(figsize = (2,2))
#     ax.imshow(array,vmin = 50, vmax = 200)
#     ax.set_title(date)
#     plt.show()
     # For each pixel I know it's latitude and longitude.
     # As you'll see below you only really need the coordinates of
     # one corner, and the resolution of the file.
     
    xmin,ymin,xmax,ymax = [xx.min(),yy.min(),xx.max(),yy.max()]
    xx = None
    yy = None
    nrows,ncols = np.shape(array)
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)   
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up), 
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
     # I don't know why rotation is in twice???
     
    output_raster = gdal.GetDriverByName('GTiff').Create(os.path.join(dir_data, 'map\dynamic_maps\lfmc\lfmc_map_%s.tif'%date),ncols, nrows, 1 ,gdal.GDT_Int16)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                # Anyone know how to specify the 
                                                 # IAU2000:49900 Mars encoding?
    output_raster.SetProjection(srs.ExportToWkt() )   # Exports the coordinate system 
                                                       # to the file
    output_raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster
    array = None
    output_raster.FlushCache()
    output_raster = None  

