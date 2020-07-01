# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:09:26 2020

@author: kkrao
"""
import pandas as pd
import os 
from dirs import dir_data, dir_codes
from pandas.tseries.offsets import DateOffset
import affine
import gdal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error


dir_lfmc = os.path.join(dir_data,'map','dynamic_maps','lfmc')                   

latlon =pd.read_csv(os.path.join(dir_data, 'reax','FuelMoistureTahoe.csv'),index_col = 'Name')
true = pd.read_excel(os.path.join(dir_data, 'reax','FuelMoistureTahoeLFMC.xlsx'),sheet_name = "Live", index_col = 0)
true.head()


pred = true.copy()

def retrieve_pixel_value(geo_coord, data_source):
    """Return floating-point value that corresponds to given point."""
    x, y = geo_coord[:,0], geo_coord[:,1]
    forward_transform =  \
        affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = (px + 0.5).astype(int), (py + 0.5).astype(int)
    pixel_coord = px, py

    data_array = np.array(data_source.GetRasterBand(1).ReadAsArray())
    return data_array[pixel_coord[0],pixel_coord[1]]



def get_pred(series):
    
    
    sm = round(series.name.day/15)
    
    date = series.name.to_period('M').to_timestamp()    
    if sm==1:
        date += DateOffset(days = 14)
    if sm==2:
        date+= DateOffset(months = 1)
        
    filename = "lfmc_map_%s.tif"%str(date.date())
    geo_coord = latlon.loc[series.index, ['X','Y']].values
    
    data_source = gdal.Open(os.path.join(dir_lfmc,filename))
    lfmc = retrieve_pixel_value(geo_coord, data_source)
    
    new_series = series.copy()
    new_series.loc[:] = lfmc
    # print(series.name)
    return new_series

pred = true.apply(get_pred, axis = 1)
pred.replace(-9999,np.nan,inplace = True)

sns.set(font_scale=1., style = 'ticks')

for col in true.columns:
    fig, ax = plt.subplots(figsize = (3,3))
    ax.scatter(true[col],pred[col])
    ax.set_title(col)
    ax.set_xlabel('True LFMC (%)')
    ax.set_ylabel('Predicted LFMC (%)')
    ax.set_xlim(50,250)
    ax.set_ylim(50,250)
    ticks = [50,100,150,200,250]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.plot(ticks,ticks,'-',color = 'lightgrey',zorder = -1)
    ax.set_aspect('equal', 'box')
    
    
    r2 = true[col].corr(pred[col])**2    
    ax.annotate('$r^2$ = %0.2f'%r2, xy=(0.1, 0.9), xycoords='axes fraction')  

fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(true.values.flatten(),pred.values.flatten(),color = 'magenta',edgecolor = 'grey')
ax.set_title('All sites')
ax.set_xlabel('True LFMC (%)')
ax.set_ylabel('Predicted LFMC (%)')
ax.set_xlim(50,250)
ax.set_ylim(50,250)
ticks = [50,100,150,200,250]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.plot(ticks,ticks,'-',color = 'lightgrey',zorder = -1)
ax.set_aspect('equal', 'box')


r2 = true.stack().corr(pred.stack())**2    
ax.annotate('$r^2$ = %0.2f'%r2, xy=(0.1, 0.9), xycoords='axes fraction') 


fig, ax = plt.subplots(figsize = (3,3))
ax.errorbar(true.mean(),pred.mean(),yerr = pred.std(), xerr = true.std(), color = 'grey',linewidth = 0, elinewidth = 1, capsize = 1,zorder =-1 )
ax.scatter(true.mean(),pred.mean(),color = 'brown',edgecolor = 'grey')
ax.set_title('All sites')
ax.set_xlabel('True $\overline{LFMC}$ (%)')
ax.set_ylabel('Predicted $\overline{LFMC}$ (%)')
ax.set_ylabel('Predicted LFMC (%)')
ax.set_xlim(50,250)
ax.set_ylim(50,250)
ticks = [50,100,150,200,250]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.plot(ticks,ticks,'-',color = 'lightgrey',zorder = -2)
ax.set_aspect('equal', 'box')


r2 = true.mean().corr(pred.mean())**2    
ax.annotate('$r^2$ = %0.2f'%r2, xy=(0.1, 0.9), xycoords='axes fraction') 