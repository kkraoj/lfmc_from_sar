# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:28:46 2019

@author: kkrao
"""
import os
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import cartopy.crs as ccrs
from osgeo import gdal, osr, gdal_array
from cartopy.feature import ShapelyFeature 
from cartopy.io.shapereader import Reader

from dirs import dir_data
from plot_functions import plot_pred_actual

#%% Plot control settings
ZOOM=1.0
FS=14*ZOOM
PPT = 0
DPI = 300
sns.set_style('ticks')
#%% fix plot dims
mpl.rcParams['font.size'] = FS
SC = 3.54331
DC = 7.48031

#%% initialize plot
lc_dict = {14: 'crop',
            20: 'crop',
            30: 'crop',
                50: 'Closed broadleaf deciduous',
            70: 'Closed needleleaf evergreen',
            90: 'Mixed forest',
            100:'Mixed forest',
            110:'Shrub/grassland',
            120:'grassland/shrubland',
            130:'Shrubland',
            140:'Grassland',
            150:'sparse vegetation',
            160:'regularly flooded forest'}
color_dict = {'Closed broadleaf deciduous':'darkorange',
              'Closed needleleaf evergreen': 'forestgreen',
              'Mixed forest':'darkslategrey',
              'Shrub/grassland' :'y' ,
              'Shrubland':'tan',
              'Grassland':'lime',
              }  
#%% functions

def corr_color_bar():
    a = np.array([[-1,1]])
    plt.figure(figsize=(0.2,3))
    plt.imshow(a, cmap="RdYlGn")
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    plt.colorbar(orientation="vertical", cax=cax, ticks = [-1,-0.5,0,0.5,1])
    # pl.savefig("colorbar.pdf")    
def scatter_lfmc_vpd():
    arr = gdal_array.LoadFile(os.path.join(dir_data,'gee/vpdmax_lfmc_2019.tif'))    
    fig, ax = plt.subplots(figsize = (SC,SC))
    plot_pred_actual(arr[1,:,:].flatten(), arr[0,:,:].flatten(),
                 xlabel = "Mean LFMC (%)", ylabel = "Mean VPD (hPa)",\
                 ax = ax,annotation = False,\
                 oneone=False,\
                 cmap = 'viridis')
    ax.set_xlim(0,150)
    ax.set_ylim(0,50)
    ax.set_aspect('auto')
    ax.set_xticks([0,50,100,150])
    ax.set_yticks([0,25,50])
def lfmc_vpd_corr_bar():
    
    df = pd.DataFrame({'corr':[ -13,12,53,9,-12,-15]}\
                       , index = [50,70,90,110,130,140])
    
    df/= 100
    df.index = df.index.map(lc_dict)
    df['Landcover'] = df.index
    fig, ax1 = plt.subplots(1,1,figsize = (SC,SC))
    
    colors = df.index.map(color_dict)
    sns.barplot(x="Landcover", y="corr", data=df, ax = ax1, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),\
                edgecolor = 'lightgrey',linewidth = 1,
                )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45,horizontalalignment='right')
    ax1.legend_.remove()
    ax1.set_xlabel('')
    ax1.set_ylabel('$R$(LFMC,VPD)')

    plt.show() 

    
def main():
    scatter_lfmc_vpd()
if __name__ == '__main__':
    main()