# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:34:06 2019

@author: kkrao
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 04:36:08 2019

@author: kkrao
"""

import pandas as pd
import numpy as np
import os
import arcpy
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import seaborn as sns
from osgeo import gdal
from skimage.transform import resize
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle




#%% load data

os.chdir(r'D:\Krishna\projects\vwc_from_radar\data\whittaker')
t = arcpy.RasterToNumPyArray(arcpy.Raster('temp.tif'),nodata_to_value=-999)
p = arcpy.RasterToNumPyArray(arcpy.Raster('ppt.tif'),nodata_to_value=-999)
e = arcpy.RasterToNumPyArray(arcpy.Raster('elevation.tif'),nodata_to_value=-999).astype(float)


t[t<0] = np.nan
p[p<0] = np.nan
e[e<0] = np.nan

e = resize(e, p.shape)
# 
##%% load sites
driver = gdal.GetDriverByName('GTiff')

training_sites = ['Blackstone',
 'Double Springs',
 'Dubois',
 'Balanced Rock',
 'Black Cedar',
 'Big Indian',
 'Fortynine C2A',
 'Cow Mountain',
 'Corralitos New Growth',
 'Blackberry Hill',
 'Juanita',
 'Deer Hill',
 'Mad River',
 'Deep Creek',
 'Keating Cutoff',
 'Jackson',
 'D06_Ute Canyon',
 'Great Divide',
 'D10 - Flagstaff Mtn. - Cold',
 'D11_Guanella_Pass',
 'D11_Miller_Gulch',
 'Blue Park',
 'D02_Red_Deer',
 'D03_Gibson',
 'D03_Willis_Creek',
 'D07_Cimarron River',
 'CNTX_Comal_TX',
 'CNTX_Hays_TX',
 'HILL_Edwards2_TX',
 'NETX_Wood_TX',
 'Mt. Baldy Village',
 'Bitter Canyon, Castaic',
 'Clark Motorway, Malibu',
 'Glendora Ridge, Glendora',
 'Cottage',
 'ASF Greer',
 'CAF Truchas 1',
 'CAF Truchas 2',
 'GNF Lincoln Canyon',
 'LNF Mayhill',
 'DUCK CREEK',
 'Lewis Canyon',
 'Hammett',
 'Kuna',
 'Morgan Creek',
 'Rock Spring',
 'Sellers Creek',
 'Ohio Gulch',
 'Rye Grass Flat',
 'Price Rec',
 'Muskrat',
 'Lost Creek, WY',
 'Reader Ranch',
 'Oak Knoll',
 'McCloud (SHF)',
 'Mad Ridge Fuel Break',
 'Rome Overlook',
 'Grass Mesa',
 'Lookout Mtn',
 'Hall Creek',
 'Sharpnose',
 'HILL_Gillespie_TX',
 'HILL_SanSaba_TX',
 'NOTX_Hood_TX',
 'NOTX_Palo_TX',
 'SETX_Newt_TX',
 'SETX_SAug_TX',
 'Lake Hughes',
 'Hastings Old Growth',
 'Mt. Woodson Station',
 'Lopez Lake',
 'Gifford',
 'Oak Flat',
 'San Marcos',
 'Los Robles, Thousand Oaks',
 'PNF Cherry',
 'PNF White Spar',
 'LNF Cosmic',
 'LNF Smokey Bear',
 'Lucky Springs',
 'Palisade',
 'Panter',
 'RUTH',
 'Simco',
 'Pocatello',
 'Sand Creek Desert',
 'Spar Canyon',
 'Table Legs Butte',
 'Sunset Heights',
 'Three Creek',
 'Squaw Peak',
 'Snowbasin',
 'TahoeDonner',
 'Tyler Foote',
 'Pulgas',
 'Saratoga Summit',
 'Weed Mill Site',
 'Shinar Saddle',
 'Six Shooter',
 'Sugar Hill',
 'Quincy RD',
 'Ziegler',
 'Sylvan',
 'Tumalo Ridge',
 'Pole Creek',
 'South Canyon',
 'Red Feather',
 'West Elk Creek',
 'RGPL_Dimmit_TX',
 'TPEC_Presidio_TX',
 'Smith Ranch',
 'Sonora',
 'Trippet Ranch, Topanga',
 'Los Alamos',
 'nacimiento',
 'ponderosa',
 'Reyes Creek',
 'Rose Valley',
 'upper oso',
 'RMV',
 'Summit2',
 'Tapo Canyon, Simi Valley',
 'Warm Springs',
 'Wells']

latlon = pd.read_csv(r'D:\Krishna\projects\vwc_from_radar\data\fuel_moisture\nfmd_queried_latlon.csv', index_col = 0)
latlon = latlon.loc[training_sites]
#drop a one row of site Fortynine C2A because of duplicate
latlon = latlon[~latlon.index.duplicated()]

def getValue(data, latlon, colname):
    latlon[colname] = np.nan
    for index, row in latlon.iterrows():
        y = int((row.Longitude - xOrigin) / pixelWidth)
        x = int((yOrigin - row.Latitude ) / pixelHeight)
        latlon.loc[index, colname] = data[x][y]
    return latlon


for filename in [ "temp.tif",  "ppt.tif",  "elevation.tif"]:
    dataset = gdal.Open(filename)
    band = dataset.GetRasterBand(1)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    transform = dataset.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    data = band.ReadAsArray(0, 0, cols, rows)
    latlon= getValue(data, latlon, filename[:-4])
#
np.save(r"D:\Krishna\projects\vwc_from_radar\data\whittaker\elevation.npy",e)
np.save(r"D:\Krishna\projects\vwc_from_radar\data\whittaker\precipitation.npy",p)
np.save(r"D:\Krishna\projects\vwc_from_radar\data\whittaker\temperature.npy",t)

latlon = latlon.drop(latlon[latlon.ppt<0].index)
latlon.to_csv(r"D:\Krishna\projects\vwc_from_radar\data\whittaker\nfmd_sites_climatology.csv")
#%% plot t vs. p
sns.set(style = 'ticks',font_scale = 1.1)

fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(t, p, c = 'grey', alpha = 0.02, edgecolor = "none")
ax.set_xlabel('Temperature ($^o$C)')
ax.set_ylabel('Precipitation (mm.yr$^{-1}$)')
ax.scatter(latlon.temp, latlon.ppt, marker = 'x', color = 'r', linewidth = 0.5)
ax.set_ylim(-150,3000)

fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(t, e, c = 'grey', alpha = 0.02, edgecolor = "none")
ax.set_xlabel('Temperature ($^o$C)')
ax.set_ylabel('Elevation (m)')
ax.scatter(latlon.temp, latlon.elevation, marker = 'x', color = 'cyan', linewidth = 0.5)

#
##%% RMSE per site
#
#df = pd.read_csv(r'D:\Krishna\projects\vwc_from_radar\data\pred_frame.csv', index_col = 0)
#def rmse(df):
#    rmse = sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat']))
#    return rmse
#latlon['rmse'] = df.groupby('site').apply(rmse)
#sns.set(style = 'ticks',font_scale = 1.5)
#
#fig, ax = plt.subplots(figsize = (4,4))
#sns.regplot(latlon.ppt, latlon.rmse, ax = ax)
#ax.set_xlabel('Precipitation (mm.yr$^{-1}$)')
#ax.set_ylabel('Site RMSE')
#
#fig, ax = plt.subplots(figsize = (4,4))
#sns.regplot(latlon.temp, latlon.rmse, ax = ax)
#ax.set_xlabel('Temperature ($^o$C)')
#ax.set_ylabel('Site RMSE')
#
#fig, ax = plt.subplots(figsize = (4,4))
#sns.regplot(latlon.elevation, latlon.rmse, ax = ax)
##ax.scatter(latlon.elevation, latlon.rmse)
#ax.set_xlabel('Elevation (m)')
#ax.set_ylabel('Site RMSE')
#
#latlon['land_cover'] = df.groupby('site')['forest_cover(t)'].min()
#
#fig, ax = plt.subplots(figsize = (4,4))
#ax.scatter(latlon.land_cover, latlon.rmse)
#ax.set_xlabel('Land cover')
#ax.set_ylabel('Site RMSE')
#
##%% Temp histogram
#
#t_hist = t.flatten()
#t_hist = t_hist[~np.isnan(t_hist)]
#
#
#
#
#latlon['examples'] = df.groupby('site').date.count()
##latlon.examples/=latlon.examples.sum()
##
#fig, ax = plt.subplots(figsize = (4,4))
##
#ax.hist(t_hist, normed = True, bins = 100)
##ax.bar(latlon.temp, 2*latlon.examples, width =0.6, color = 'r', alpha = 0.5)
#ax.set_xlabel('Temperature ($^o$C)')
#ax.set_ylabel('Frequency')
##
##(latlon.rmse**2*latlon.examples).sum()**0.5
#
#
##%% weighted rmse calc
#
#latlon.drop_duplicates(inplace = True)
#latlon.dropna(inplace = True)
#latlon['se'] = latlon.rmse**2*latlon.examples
#
#def get_site_pdf(df,col_name = 'temp'):
#    col = np.repeat(df[col_name].values,df['examples'].astype(int).values)
#    y,x = np.histogram(col,bins=100, density = True)
#    x = x[:-1]
#    df[col_name+'_hist'] = np.nan
#    for index, row in df.iterrows():
#        df.loc[index,col_name+'_hist'] = y[x[x<=df[col_name].loc[index]].argmax()]
#    df[col_name+'_hist']/=df[col_name+'_hist'].sum() #normalize prability mass function
#    return df
#
#latlon = get_site_pdf(latlon, col_name = 'temp')
#latlon = get_site_pdf(latlon, col_name = 'land_cover')
#latlon = get_site_pdf(latlon, col_name = 'elevation')
#latlon = get_site_pdf(latlon, col_name = 'ppt')
#
#def get_roi_pdf(df):
#    vars = ['temp','ppt','elevation']
#    ctr = 0
#    for var in [t, p,e]:
#        hist = var.flatten()[~np.isnan(var.flatten())]
#        y,x = np.histogram(hist,bins=100, density = True)
#        x = x[:-1]
#        df[vars[ctr]+'_roi_hist'] = np.nan
#        for index, row in df.iterrows():
#            df.loc[index,vars[ctr]+'_roi_hist'] = y[x[x<=df[vars[ctr]].loc[index]].argmax()] 
#        df[vars[ctr]+'_roi_hist']/=df[vars[ctr]+'_roi_hist'].sum() #normalize prability mass function
#        ctr+=1
#    return df
#
#latlon = get_roi_pdf(latlon)
#
##%% sample overlaying hist
#
#alpha = 0.5
#fig, ax = plt.subplots(figsize = (5,5))
#ax.bar(latlon.temp.values, latlon.temp_hist.values, alpha = alpha, color = 'b', label = 'sites')
#ax.bar(latlon.temp.values, latlon.temp_roi_hist.values, alpha = alpha, color = 'm', label = 'ROI')
#ax.set_xlabel('MAT ($^o$C)')
#ax.set_ylabel('pdf')
#plt.legend()
#
#alpha = 0.5
#fig, ax = plt.subplots(figsize = (5,5))
#ax.bar(latlon.elevation.values, latlon.elevation_hist.values, alpha = alpha, color = 'b', label = 'sites', width = 100)
#ax.bar(latlon.elevation.values, latlon.elevation_roi_hist.values, alpha = alpha, color = 'm', label = 'ROI', width = 100)
#ax.set_xlabel('Altitude (m)')
#ax.set_ylabel('pdf')
#plt.legend()
#
#alpha = 0.5
#fig, ax = plt.subplots(figsize = (5,5))
#ax.bar(latlon.ppt.values, latlon.ppt_hist.values, alpha = alpha, color = 'b', label = 'sites', width = 100)
#ax.bar(latlon.ppt.values, latlon.ppt_roi_hist.values, alpha = alpha, color = 'm', label = 'ROI', width = 100)
#ax.set_xlabel('MAP (mm.yr$^{-1}$)')
#ax.set_ylabel('pdf')
#plt.legend()
#
#
##%% RMSE_w
#RMSE =    ((latlon.se).sum()/(latlon.examples.sum()))**0.5
#RMSE_w_T =  ((latlon.se*latlon.temp_roi_hist/latlon.temp_hist).sum()/(latlon.examples.sum()))**0.5
#RMSE_w_E =  ((latlon.se*latlon.elevation_roi_hist/latlon.elevation_hist).sum()/(latlon.examples.sum()))**0.5
#RMSE_w_P =  ((latlon.se*latlon.ppt_roi_hist/latlon.ppt_hist).sum()/(latlon.examples.sum()))**0.5
#
#print('RMSE weighted by precipitation = %0.2f %%'%RMSE_w_P)
#print('RMSE weighted by elevation = %0.2f %%'%RMSE_w_E)
#print('RMSE weighted by temperature = %0.2f %%'%RMSE_w_T)