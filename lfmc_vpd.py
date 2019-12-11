# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:28:46 2019

@author: kkrao
"""
import os
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

import cartopy.crs as ccrs
from osgeo import gdal, osr, gdal_array
from cartopy.feature import ShapelyFeature 
from cartopy.io.shapereader import Reader

from dirs import dir_data
from plot_functions import plot_pred_actual

#%% Plot control settings
ZOOM=1
FS=12*ZOOM
PPT = 0
DPI = 300
sns.set_style('ticks')
#%% fix plot dims
mpl.rcParams['font.size'] = FS
mpl.rcParams['axes.titlesize'] = 'medium'
SC = 3.54331*ZOOM
DC = 7.48031*ZOOM

#%% initialize plot
lc_dict = {14: 'crop',
            20: 'crop',
            30: 'crop',
                50: 'Closed broadleaf\ndeciduous',
            70: 'Closed needleleaf\nevergreen',
            90: 'Mixed forest',
            100:'Mixed forest',
            110:'Shrub/grassland',
            120:'grassland/shrubland',
            130:'Shrubland',
            140:'Grassland',
            150:'sparse vegetation',
            160:'regularly flooded forest'}
color_dict = {'Closed broadleaf\ndeciduous':'darkorange',
              'Closed needleleaf\nevergreen': 'forestgreen',
              'Mixed forest':'darkslategrey',
              'Shrub/grassland' :'y' ,
              'Shrubland':'tan',
              'Grassland':'lime',
              }  
SEED = 1
np.random.seed(SEED)
#%% functions

def corr_color_bar():
    a = np.array([[-1,1]])
    plt.figure(figsize=(0.2,3))
    plt.imshow(a, cmap="RdYlGn")
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    plt.colorbar(orientation="vertical", cax=cax, ticks = [-1,-0.5,0,0.5,1])
    # pl.savefig("colorbar.pdf")    
def sample(x,y,sampling_ratio = 10000):
    non_nan_ind=np.where(~np.isnan(x))[0]
    x=x.take(non_nan_ind);y=y.take(non_nan_ind)
    non_nan_ind=np.where(~np.isnan(y))[0]
    x=x.take(non_nan_ind);y=y.take(non_nan_ind)
    nsamples = int(len(x)/sampling_ratio)
    inds = np.random.choice(len(x), nsamples, replace = False)
    x=x.take(inds);y=y.take(inds)
    return x,y   
def scatter_lfmc_vpd_by_lc():
    arr = gdal_array.LoadFile(os.path.join(dir_data,'gee/vpdmax_lfmc_lc_2019_500.tif'))    
    ## subsetting for law VPD locations only
    arr[0,arr[0,:,:]>25] = np.nan
    x = arr[1,:,:].flatten()
    y = arr[0,:,:].flatten()
    xarr = arr[1,:,:].flatten()
    yarr = arr[0,:,:].flatten()
    lcarr = arr[2,:,:].flatten()
    sampling_ratio = 100

    fig, axs = plt.subplots(2,3,figsize = (DC,0.67*DC),sharex = True, sharey= True)
    
    for lc, ax in zip([50,70,90,110,130,140],axs.reshape(-1)):
        x = xarr[lcarr==lc]
        y = yarr[lcarr==lc]/10
        # cmap = mpl.colors.LinearSegmentedColormap.\
        #           from_list("", ["w",color_dict[lc_dict[lc]]])
        cmap = sns.cubehelix_palette(rot = -0.4,as_cmap = True)
        x,y = sample(x,y,sampling_ratio = sampling_ratio)
        sns.kdeplot(x,y,cmap = cmap, shade = True, legend = False, ax = ax, shade_lowest = False)
        # plot_pred_actual(x,y,
        #               xlabel = "Mean LFMC (%)", ylabel = "Mean VPD (hPa)",\
        #               ax = ax,annotation = False,\
        #               oneone=False,\
        #               cmap = ListedColormap(sns.cubehelix_palette(rot = -0.4).as_hex()))
        non_nan_ind=np.where(~np.isnan(x))[0]
        x=x.take(non_nan_ind);y=y.take(non_nan_ind)
        slope, intercept, r_value, p_value, std_err =\
            stats.linregress(x,y)
        xs = np.linspace(50,125)
        ys = slope*xs+intercept
        ax.plot(xs,ys,color = 'k', lw = 1)
        
        ax.set_xlim(50,125)
        ax.set_ylim(0,5)
        # ax.set_aspect('auto')
        ax.set_xticks([50,75,100,125])
        ax.set_yticks([0,2.5,5])
        # ax.set_xlabel('Mean LFMC (%)')
        # ax.set_ylabel('Mean VPD (hPa)')
        # ax.invert_xaxis()
        # print('p value = %0.3f'%p_value)
        
        ax.annotate('$R$ = %0.2f'%r_value, \
                    xy=(0.95, 0.95), xycoords='axes fraction',\
                    ha='right',va='top')
        ax.set_title(lc_dict[lc])
    axs[1,1].set_xlabel('Mean LFMC (%)')
    axs[1,0].annotate('Mean VPD (kPa)'%slope, \
                    xy=(-0.25, 1.1), xycoords='axes fraction',\
                    ha='right',va='center',rotation = 'vertical')
    plt.show()
    
    #############################################################
    fig, ax = plt.subplots(figsize = (DC*0.3, DC*0.3))
    x = xarr
    y =yarr/10
    

    cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0.05, light=.95,as_cmap = True)

    x,y = sample(x,y,sampling_ratio = sampling_ratio)
    sns.kdeplot(x,y,cmap = cmap, shade = True, legend = False, ax = ax,\
                shade_lowest = False)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    xs = np.linspace(50,125)
    ys = slope*xs+intercept
    ax.plot(xs,ys,color = 'k', lw = 1)

    ax.set_xlim(50,125)
    ax.set_ylim(0,5)
    ax.set_aspect('auto')
    ax.set_xticks([50,75,100,125])
    ax.set_yticks([0,2.5,5])
    ax.set_xlabel('Mean LFMC (%)')
    ax.set_ylabel('Mean VPD (kPa)')
    # ax.invert_xaxis()
    ax.annotate('$R$ = %0.2f'%r_value, \
                    xy=(0.95, 0.95), xycoords='axes fraction',\
                    ha='right',va='top')
    ax.set_title('Overall')
    plt.show()
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
    scatter_lfmc_vpd_by_lc()
if __name__ == '__main__':
    main()