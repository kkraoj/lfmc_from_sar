# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 00:11:14 2017

@author: kkrao
"""

import os
import numpy as np
import pandas as pd
import pickle
import numpy.ma as ma

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon, Patch

import cartopy.crs as ccrs
from osgeo import gdal, osr
from cartopy.feature import ShapelyFeature 
from cartopy.io.shapereader import Reader

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
from skmisc.loess import loess,loess_model
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats




from dirs import dir_data, dir_codes, dir_figures
from fnn_smoothed_anomaly_all_sites import plot_importance, plot_usa
from QC_of_sites import clean_fmc

#mpl.rcParams.update({'figure.autolayout': True})
#


#%% Plot control settings
ZOOM=1.0
FS=10*ZOOM
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

pkl_file = open(os.path.join(dir_data,'encoder.pkl'), 'rb')
encoder = pickle.load(pkl_file) 
pkl_file.close()

#%% RMSE vs sites. bar chart

frame = pd.read_csv(os.path.join(dir_data,'model_predictions_all_sites.csv'))

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
def bar_chart():
    rmse = pd.DataFrame({'RMSE':frame.groupby('site').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat']))).sort_values()})
    rmse['Landcover'] = frame.groupby('site').apply(lambda df: df['forest_cover(t)'].astype(int).values[0])
    rmse['Landcover'] = encoder.inverse_transform(rmse['Landcover'].values)
    rmse['Landcover'] = rmse['Landcover'].map(lc_dict)
    rmse['ubrmse'] = frame.groupby('site').apply(\
      lambda df: ubrmse(df['percent(t)'],df['percent(t)_hat']))
    ### mean rmse values store
    lc_rmse = frame.groupby('forest_cover(t)').apply(\
        lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat'])))
    lc_rmse.index = encoder.inverse_transform(lc_rmse.index.astype(int))
    lc_rmse.index =  lc_rmse.index.map(lc_dict)
    lc_rmse.index.name = 'Landcover'
    lc_rmse.name = 'lc_rmse'
    rmse = pd.merge(rmse, lc_rmse, on = 'Landcover')
    
    ### sort by landcover then ascending
    lc_order = lc_rmse.sort_values()
    lc_order.loc[:] = range(len(lc_order))
    rmse['lc_order'] = rmse['Landcover'].map(lc_order)
    rmse.sort_values(by=['lc_order','RMSE'], inplace = True)
    rmse['Sites'] = range(len(rmse))
    
    
    count = (rmse['ubrmse']<=0.5*rmse['RMSE']).mean()
    
    fig, _ = plt.subplots(2,3, figsize = (DC,4))
    grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.15, figure = fig)
    ax1 = plt.subplot(grid[0:, 0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, 1])
    ax4 = plt.subplot(grid[2, 1])       
    colors = lc_order.index.map(color_dict)
    sns.barplot(x="Sites", y="RMSE", data=rmse, ax = ax1, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),\
                edgecolor = 'w',linewidth = 0.04,
                )
    ## plot mean rmse points
    # ax1.plot(rmse.Sites, rmse.lc_rmse, '-',color='darkgrey',alpha = 0.8,\
    #          label = 'Overall landcover RMSE')
    ## plot ubrmse
    ax1.plot(rmse.Sites, rmse.ubrmse,'D',color = 'k',ms = 1,\
             label = 'ubRMSE',zorder = 2,mew = 0)
   
    # ax1.set_xticklabels(range(len(rmse)))
    ax1.set_xticks([])
    ax1.set_yticks([0,20,40,60,80])
    ax1.set_ylabel('RMSE (%)')
    # ax1.set_xticklabels([0,25,50,75,100,124])
    change_width(ax1, 1)
    ax1.set_ylim(0,80)

    ##### plotting 
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(handles.pop(0))
    labels.append(labels.pop(0))
    # handles.append(handles.pop(0))
    # labels.append(labels.pop(0))
    ax1.legend(handles, labels, loc = 'upper left',prop={'size': 7},frameon = False)
    
    #%% timeseries for three sites
    new_frame = frame.copy()
    new_frame.index = pd.to_datetime(new_frame.date)
    rmse = new_frame.groupby('site').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat']))).sort_values()

    for site, ax in zip([8,63,-8], [ax2, ax3, ax4]):
        sub = new_frame.loc[new_frame.site == rmse.index[site]]

        lc = lc_dict[encoder.inverse_transform(sub['forest_cover(t)'].\
                                                   astype(int).unique()[0])]
        if ax == ax4:       
            sub.plot(y = 'percent(t)', linestyle = '-', markeredgecolor = 'k', ax = ax,\
                marker = 'o', label = 'observed', color = 'grey', mew =0.3,ms = 3,linewidth = 1 )
        else:
            sub.plot(y = 'percent(t)', linestyle = '-', markeredgecolor = 'k', ax = ax,\
                marker = 'o', label = '_nolegend_', color = 'grey', mew =0.3,ms = 3,linewidth = 1 )

        sub.plot(y = 'percent(t)_hat', linestyle = '--', markeredgecolor = 'k', ax = ax,\
                marker = 'o', label = 'estimated',color = color_dict[lc], mew= 0.3, ms = 3, lw = 1, rot = 0)
        if ax==ax3:
            ax.set_ylabel('LFMC(%)') 
        ax.set_xlabel('')
        ax.legend(prop ={'size':7})
        #set ticks every week
        # ax.xaxis.set_major_locator(mdates.YearLocator())
        #set major ticks format
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%yyyy'))


        ax.set_xlim(pd.to_datetime(['Jul-2015','31-Dec-2018']))
        # ax.xaxis.grid(True, which="major", linestyle='--')
        ax.xaxis.set_minor_locator(mdates.MonthLocator(
                                                interval=6))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        # ax.xaxis.grid(True, which="minor")
        # ax.yaxis.grid()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
        # ax.set_xticklabels(ha='center')
        plt.sca(ax)
        plt.xticks(ha='center')    
        ax.tick_params(axis='both', which='minor', labelsize=FS - 2)
        if ax == ax2 or ax == ax3:
            plt.setp(ax.get_xmajorticklabels(), visible=False)
            plt.setp(ax.get_xminorticklabels(), visible=False)   
    #name panels
    ax1.annotate('a.', xy=(-0.15, 1), xycoords='axes fraction',\
                 ha='right',va='bottom', weight = 'bold') 
    ax2.annotate('b.', xy=(-0.28, 1), xycoords='axes fraction',\
                 ha='right',va='bottom', weight = 'bold')
    ax3.annotate('c.', xy=(-0.28, 1), xycoords='axes fraction',\
                 ha='right',va='bottom', weight = 'bold') 
    ax4.annotate('d.', xy=(-0.28, 1), xycoords='axes fraction',\
                 ha='right',va='bottom', weight = 'bold') 
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'bar_plot.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show()       
def ubrmse(true,pred):
    return np.sqrt(mean_squared_error(true-true.mean(),pred-pred.mean()))  
def hatching(patches,hatch = '/'):
    for i, bar in enumerate(patches):
        bar.set_hatch(hatch)
#%% performance by landcover table
def landcover_table():
    groupby = 'forest_cover(t)'
    frame = pd.read_csv(os.path.join(dir_data,'model_predictions_all_sites.csv'))

    frame = frame.loc[frame[groupby]==0]
    groupby = 'site'
    table = pd.DataFrame({'RMSE':frame.groupby(groupby).apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat'])))}).round(1)
    table['R2'] = frame.groupby(groupby).apply(lambda df: r2_score(df['percent(t)'],df['percent(t)_hat'])).round(2)
    table['N_obs'] = frame.groupby(groupby).apply(lambda df: df.shape[0])
    table['N_sites'] = frame.groupby(groupby).apply(lambda df: len(df.site.unique()))
    table['MBE'] = frame.groupby(groupby).apply(lambda df: (df['percent(t)'] - df['percent(t)_hat']).mean()).round(1)
    table['SD'] = frame.groupby(groupby).apply(lambda df: df['percent(t)'].std().round(1))
    table['Mean'] = frame.groupby(groupby).apply(lambda df: df['percent(t)'].mean().round(1))

    ### works only with original encoder!!
    
    # table.index = encoder.inverse_transform(table.index.astype(int))
    # table.index = table.index.map(lc_dict)
    # table.index.name = 'landcover'
    table = table[['N_sites','N_obs','SD','RMSE','R2','MBE','Mean']]
    table.sort_values('R2',inplace = True)
    overall = table.sum()
    overall.name = 'Overall'
    overall['SD'] = 41.6
    overall['RMSE'] = 25.0
    overall['R2'] = 0.63
    overall['MBE'] = 1.9
    table = table.append(overall)
    print(table)
    # table.to_excel('model_performance_by_lc.xls')
    # table.to_latex(os.path.join(dir_figures,'model_performance_by_lc.tex'))

#%% prediction scatter plots after CV
def plot_pred_actual(test_y, pred_y, weights = None, \
                     cmap = ListedColormap(sns.cubehelix_palette().as_hex()), \
                     axis_lim = [-25,50],\
                 xlabel = "None", ylabel = "None", ticks = None,\
                 ms = 8, mec ='', mew = 0, ax = None,annotation = True,\
                 oneone=True):
    # plt.axis('scaled')
    ax.set_aspect('equal', 'box')

    x = test_y
    y = pred_y

        
    non_nan_ind=np.where(~np.isnan(x))[0]
    x=x.take(non_nan_ind);y=y.take(non_nan_ind)
    non_nan_ind=np.where(~np.isnan(y))[0]
    x=x.take(non_nan_ind);y=y.take(non_nan_ind)

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plot = ax.scatter(x,y, c=z, s=ms, edgecolor=mec, cmap = cmap, linewidth = mew)
    if oneone:
        ax.plot(axis_lim,axis_lim, lw =.2, color = 'grey')
    
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yticks(ax.get_xticks())
#    ax.set_xticks([-50,0,50,100])
#    ax.set_yticks([-50,0,50,100])

    R2 = r2_score(x,y,sample_weight = weights)
    model_rmse = np.sqrt(mean_squared_error(x,y))
    model_bias = np.mean(pred_y - test_y)
    if annotation:
        ax.annotate('$R^2=%0.2f$\nRMSE = %0.1f%%\nBias = %0.1f%%'%(np.floor(R2*100)/100, model_rmse, np.abs(model_bias)), \
                    xy=(0.03, 0.97), xycoords='axes fraction',\
                    ha='left',va='top')
    if axis_lim:
        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    # plt.tight_layout()      
def scatter_plot_all_3():
    fig, _ = plt.subplots(1,3, figsize = (DC,DC/3))
    grid = plt.GridSpec(1,3, wspace=0.6, figure = fig)
    
    ax1 = plt.subplot(grid[0,0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[0,2])
    ##############################

    x = frame['percent(t)'].values
    y = frame['percent(t)_hat'].values
    
    plot_pred_actual(x, y,ax = ax1,ticks = [0,100,200,300],\
        ms = 40, axis_lim = [0,300], xlabel = "$LFMC_{obs}$ (%)", \
        ylabel = "$LFMC_{est}$ (%)",mec = 'grey', mew = 0)
    
    ##############################
    t = frame.groupby('site')['percent(t)','percent(t)_hat'].mean()
    x = t['percent(t)'].values
    y = t['percent(t)_hat'].values
    weights = frame.groupby('site').apply(lambda df: len(df['percent(t)']))
    plot_pred_actual(x, y,ax = ax2,\
                 ms = 40, axis_lim = [50,200], xlabel = "$\overline{LFMC_{obs}}$ (%)", \
            ylabel = "$\overline{LFMC_{est}}$ (%)",mec = 'grey', \
            mew = 0.3, ticks = [50,100,150,200],weights = weights)
    n = np.abs(x-y)/x
    n = n[n>=0.2]
    n = len(n)/len(x)*100
    print('[INFO] %d%% sites had bias >= 20%%'%np.round(n))
    ##############################
    ndf = frame[['site','date','percent(t)','percent(t)_hat']]
    
    x = ndf.groupby(['site']).apply(lambda x: (x['percent(t)'] - x['percent(t)'].mean())).values
    y = ndf.groupby(['site']).apply(lambda x: (x['percent(t)_hat'] - x['percent(t)_hat'].mean())).values
    
    plot_pred_actual(x, y,ax = ax3,ticks = [-100,-50,0,50,100],\
                  ms = 40, axis_lim = [-100,100], xlabel = "$LFMC_{obs} - \overline{LFMC_{obs}}$ (%)", \
            ylabel = "$LFMC_{est} - \overline{LFMC_{est}}$ (%)",mec = 'grey', mew = 0)
    ##############################
    df = pd.DataFrame({'$R^2_{test}$':frame.groupby('site').apply(lambda df: r2_score(df['percent(t)'],df['percent(t)_hat'])).sort_values()})
    df['site_mean_error'] = frame.groupby('site').apply(lambda df: df['percent(t)'].mean() - df['percent(t)_hat'].mean())
    df['site_mean'] = frame.groupby('site').apply(lambda df: df['percent(t)'].mean())
    df['RMSE'] = frame.groupby('site').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat'])))
    df['Landcover'] = frame.groupby('site').apply(lambda df: df['forest_cover(t)'].astype(int).values[0])
    df['Landcover'] = encoder.inverse_transform(df['Landcover'].values)
    df['Landcover'] = df['Landcover'].map(lc_dict)
    df['colors'] = df['Landcover'].map(color_dict)
    df['true_sd'] = frame.groupby('site').apply(lambda df: df['percent(t)'].std())
    df['CV'] = frame.groupby('site').apply(lambda df: df['percent(t)'].std()/df['percent(t)'].mean())
    df['R'] = frame.groupby('site').apply(lambda df: np.corrcoef(df['percent(t)'],df['percent(t)_hat'])[0,1])
    df['N_obs'] = frame.groupby('site').apply(lambda df: len(df['percent(t)']))
    df.sort_values(by = 'N_obs',inplace = True, ascending = False)
    
    df = frame.groupby('site')['percent(t)','percent(t)_hat'].mean()
    df = frame.groupby(['forest_cover(t)','site']).apply(lambda \
            df:(df['percent(t)']-df['percent(t)_hat']).mean()/df['percent(t)'].mean()).reset_index()
    df.columns = ['landcover','site','bias']
    df =  df.groupby('landcover').apply(lambda df: (df['bias'].abs()>=0.2).mean())
    df.index = encoder.inverse_transform(df.index.astype(int).values)
    df.index = df.index.map(lc_dict)
    df.sort_values(inplace = True)
    
    # Hide the right and top spines
    for ax in [ax1,ax2,ax3]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ##############################
    ax1.annotate('a.', xy=(-0.28, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')  
    ax2.annotate('b.', xy=(-0.28, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')      
    ax3.annotate('c.', xy=(-0.28, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'scatter_plot_all_3.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    
        
    plt.show()   
def microwave_importance():
    
    
    
    purple = '#7570b3'
    green = '#1b9e77'
    
    fig, _ = plt.subplots(2,2, figsize = (SC,SC))
    grid = plt.GridSpec(2,2, wspace=0,hspace= 0, figure = fig)
    
    ax1 = plt.subplot(grid[0,0])
    ax2 = plt.subplot(grid[1, 1])
    ax3 = plt.subplot(grid[1, 0]) 
    ax4 = plt.subplot(grid[0, 1]) 
    # fig, ax = plt.subplots(figsize = (2,0.75))
    # Example data
    
    #####################################################
    people = ( 'Without\nMicrowave','With\nMicrowave')
    y_pos = np.arange(len(people))
    performance = [ 0.61,0.73]
    
    rects = ax2.barh(y_pos, performance,height = 0.5, align='center', color = [green,purple])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(people)
    ax2.xaxis.tick_top()
    ax2.set_xlim(0,1)
    ax2.set_ylim(-1,2)
    ax2.set_xticks([0.0,0.5,1])
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    # ax2.annotate('0.0', xy=(0.03, 1.07), xycoords='axes fraction',\
    #             ha='left',va='bottom')
    # Shift the text to the left side of the right edge
    xloc =  5
    # White on magenta
    clr = 'white'
    align = 'right'
    ctr = 0    
    for rect in rects:
        # Center the text vertically in the bar
        width = int(rect.get_width())
        yloc = rect.get_y() + rect.get_height() / 2
        ax2.annotate('$R^2$=%0.2f'%performance[ctr], xy=(width, yloc), xytext=(performance[ctr]*100-xloc, 0),
                            textcoords="offset points",
                            ha=align, va='center',
                            color=clr, clip_on=False)
        ctr+=1

    ################################################################
    performance = [28.0,25.0]
    rects = ax1.barh(y_pos, performance,height = 0.5, align='center',  color = [green,purple])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(people)
    ax1.yaxis.tick_right()
    ax1.set_xlim(30,0)
    ax1.set_ylim(-1.1,2)
    ax1.set_xticks([10,20,30])
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # ax1.annotate('0', xy=(0.97 ,-0.07), xycoords='axes fraction',\
    #             ha='right',va='top')
    # Shift the text to the left side of the right edge
    xloc = 5
    # White on magenta
    align = 'left'
    ctr = 0    
    for rect in rects:
        # Center the text vertically in the bar
        width = int(rect.get_width())
        yloc = rect.get_y() + rect.get_height() / 2
        ax1.annotate('RMSE=%0.1f'%performance[ctr], xy=(width, yloc), xytext=(xloc, 0),
                            textcoords="offset points",
                            ha=align, va='center',
                            color=clr, clip_on=True)
        ctr+=1
    
    fig.delaxes(ax3)
    fig.delaxes(ax4)
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'microwave_importance`.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show() 
def nfmd_sites():
    ################################################################################
    
    
    fig = plt.figure(constrained_layout=True,figsize=(0.67*DC, 3))
    widths = [2, 1]
    heights = [1,1]
    spec = mpl.gridspec.GridSpec(ncols=2, nrows=2, width_ratios=widths,\
                                 height_ratios = heights,\
                                 wspace = 0.4,hspace = 0.15)
    
    ax1 = plt.subplot(spec[0:,0])
    ax2 = plt.subplot(spec[0,1])
    ax3 = plt.subplot(spec[1,1])
    
    
    # fig, _ = plt.subplots(2,3,figsize=(0.67*DC, SC), constrained_layout = True)
    # cols = 5
    # grid = plt.GridSpec(2,cols, wspace=-0,hspace= 0.2  , figure = fig)
    
    # ax1 = plt.subplot(grid[0:,0:cols-1])
    # ax2 = plt.subplot(grid[0, cols-1])
    # ax3 = plt.subplot(grid[1, cols-1]) 
    
    dir_data = "D:/Krishna/projects/vwc_from_radar/data/fuel_moisture"
    os.chdir(dir_data)
    files = os.listdir(dir_data+'/raw/')
    Df = pd.DataFrame()
    for file in files:
        Df = pd.concat([Df, pd.read_table('raw/'+file)])
    Df.drop("Unnamed: 7", axis = 1, inplace = True)
    Df["Date"] = pd.to_datetime(Df["Date"])
    
    Df['address'] = Df["Site"] + ', '+ Df["State"]
    pd.DataFrame(Df['address'].unique(), columns = ['address']).to_csv('address.csv')
    
    ## actual latlon queried from nfmd
    latlon = pd.read_csv("nfmd_queried_latlon.csv", index_col = 0)
    latlon['observations'] = Df.groupby('Site').GACC.count()
    latlon.rename(columns = {"Latitude":"latitude", "Longitude":"longitude"}, inplace = True)
    temp = Df.drop_duplicates(subset = 'Site')
    temp.index = temp.Site
    latlon['State'] = np.NaN
    latlon.update(temp)
     
    ##############################################################################
    ### plot of data record
    os.chdir('D:/Krishna/projects/vwc_from_radar')
    df = pd.read_pickle('data/df_sar_vwc_all')
    df.residual = df.residual.abs()
    # df = df.loc[df.residual<=2, :]
    #df = df.loc[df.data_points>=10, :]
    latlon = pd.read_csv('data/fuel_moisture/nfmd_spatial.csv', index_col = 0)
    ### import 50 sites
    # selected_sites = pd.read_pickle('data/lstm_input_data_pure+all_same_28_may_2019_res_SM_gap_3M').site.unique()
    selected_sites = list(frame.site.unique())
    selected_sites.remove('Baker Park') # This is in south dakota
    latlon['color'] = 'lightgrey'
    latlon.loc[selected_sites,'color'] = 'maroon'
    latlon = latlon.loc[selected_sites] # plotting only red sites for IGARSS
    latlon.sort_values('color', inplace = True)
    # latlon['data_points'] = df.groupby('Site').obs_date.count()
    latlon['data_points'] = frame.groupby('site').date.count()
    latlon.sort_values('data_points',inplace = True, ascending = False)
    #latlon = latlon.loc[latlon.data_points>=10,:]
    cmap = 'magma'
    sns.set_style('ticks')
    alpha = 1
    
    m = Basemap(llcrnrlon=-119,llcrnrlat=22.8,urcrnrlon=-92,urcrnrlat=52,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95, ax = ax1)
    m.drawmapboundary(fill_color='lightcyan',zorder = 0.9)
    #-----------------------------------------------------------------------
    # load the shapefile, use the name 'states'
    m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                    name='states', drawbounds=True)
    statenames=[]
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        statenames.append(statename)
    for nshape,seg in enumerate(m.states): 
        poly = Polygon(seg,facecolor='papayawhip',edgecolor='k', zorder  = 2,alpha = 0.6)
        ax1.add_patch(poly)
    ### adding ndvi
    plot=m.scatter(latlon.longitude.values, latlon.latitude.values, 
                   s=2*latlon.data_points,c=latlon.color.values,edgecolor = 'lightgrey',linewidth = 0.5,\
                        marker='o',alpha = 0.9,latlon = True, zorder = 4,\
                        )
    #################
    ds = gdal.Open(r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa_ndvi.tif")
    data = ds.ReadAsArray()
    data = np.ma.masked_where(data<=0,data)
    x = np.linspace(-134.4, -134.4+0.1*data.shape[1]-0.1,data.shape[1]) #from raster.GetGeoTransform()
    y  = np.linspace(52.0,52.0-0.1*data.shape[0]+0.1,data.shape[0]) 
    xx, yy = np.meshgrid(x, y)  
    m.pcolormesh(xx, yy, data, cmap = 'YlGn',vmin =0,vmax = 250,\
                 zorder = 1,latlon=True) 
    m1 = ax1.scatter([],[], color = 'maroon',edgecolor = 'lightgrey',\
                     linewidth = 0.5, s=2*10)
    m2 = ax1.scatter([],[], color = 'maroon',edgecolor = 'lightgrey',\
                     linewidth = 0.5, s=2*50)
    m3 = ax1.scatter([],[], color = 'maroon',edgecolor = 'lightgrey',\
                     linewidth = 0.5, s=2*100)
    legend_markers = [m1, m2,m3]

    labels =['10','50','100']

    legend = ax1.legend(handles=legend_markers, labels=labels, loc='upper center',
        scatterpoints=1,title = 'No. of Measurements',ncol = 3,\
        handletextpad=0,bbox_to_anchor = (0.5,0.01),fontsize = 'small',borderpad = 0.5)
    plt.setp(legend.get_title(),fontsize='small')
    ###############
    m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                    name='states', drawbounds=True)
    #ax.set_title('Length of data record (number of points $\geq$ 10)')
    plt.setp(ax1.spines.values(), color='w')
    #plt.legend()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.08)
    #cb=fig.colorbar(plot,ax=ax,cax=cax, ticks = np.linspace(0,20,5))
    #cax.annotate('$\Delta$ days',xy=(0,1.0), xycoords='axes fraction',\
    #            ha='left')
    # plt.savefig(os.path.join(dir_figures,'nfmd_sites'), dpi =600,\
                        # bbox_inches="tight")
    e = np.load(r"D:\Krishna\projects\vwc_from_radar\data\whittaker\elevation.npy")
    p = np.load(r"D:\Krishna\projects\vwc_from_radar\data\whittaker\precipitation.npy")
    t = np.load(r"D:\Krishna\projects\vwc_from_radar\data\whittaker\temperature.npy")

    latlon = pd.read_csv(r"D:\Krishna\projects\vwc_from_radar\data\whittaker\nfmd_sites_climatology.csv",\
                         index_col = 0)
    ax2.hexbin(t.flatten(),p.flatten(),cmap ='Greys',gridsize = (20,14),\
               linewidths=0.1,norm=mpl.colors.LogNorm(vmin=0.8, vmax=10000))
    ax2.set_xlim(-1,25)
    ax2.set_ylim(-200,4200)
    ax2.set_ylabel('MAP (mm.yr$^{-1}$)')
    ax2.set_xticks([0,10,20])
    ax2.set_xticklabels([])
    ax2.set_ylim(top = 4200)

    ax2.scatter(latlon.temp, latlon.ppt, marker = 'o', color = 'maroon',\
                s = 8,linewidth = 0.5,edgecolor = 'w',label = '_nolegend_')
    # ax2.set_ylim(-150,3000)

    ax3.hexbin(t.flatten(),e.flatten(),cmap ='Greys',gridsize = (20,14),\
               linewidths=0.1,norm=mpl.colors.LogNorm(vmin=1, vmax=10000),label = '_nolegend_')
    ax3.set_xlim(-1,25)
    ax3.set_ylim(-200,4200)
    ax3.set_xlabel('MAT ($^o$C)')
    ax3.set_ylabel('Elevation (m)')
    ax3.scatter(latlon.temp, latlon.elevation, marker = 'o', color = 'maroon',\
                s = 8,linewidth = 0.5,edgecolor = 'w', label = 'Sites')
    ax3.set_xticks([0,10,20])
    
    
    ax1.annotate('a.', xy=(-0.0, 1.02), xycoords='axes fraction',\
                ha='left',va='bottom', weight = 'bold')  
    ax2.annotate('b.', xy=(-0.6, 1.04), xycoords='axes fraction',\
                ha='left',va='bottom', weight = 'bold')      
    ax3.annotate('c.', xy=(-0.6, 1.04), xycoords='axes fraction',\
                ha='left',va='bottom', weight = 'bold')
    
    
    ax3.scatter([],[],label = 'West USA',marker = 'h',s=10,color = 'grey')    
    ax3.legend(prop={'size': FS-3},frameon = False,handletextpad =-0.3,\
               borderpad=0.2,bbox_to_anchor = (1.02,1.02))
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'sites.jpg'), \
                                 dpi =DPI, bbox_inches="tight")
    

    plt.show()
    print('[INFO] Mean MAP in West USA = %d mm/yr, in sites = %d mm/yr'%(np.nanmean(p), latlon.ppt.mean()))
    print('[INFO] Mean MAT in West USA = %d C, in sites = %d C'%(np.nanmean(t), latlon.temp.mean()))
    print('[INFO] Mean Elevation in West USA = %d m, in sites = %d m'%(np.nanmean(e), latlon.elevation.mean()))
def rmsd(df):
    df.index = df.date
    df = df.groupby('fuel').resample('1d').asfreq()['percent'].interpolate()
    df = df.reset_index()
    df.index = df.date
    df = df.pivot(values = 'percent', columns = 'fuel')
    corr = df.corr().min().min()
    return corr
def sites_QC():
    df = pd.read_pickle(os.path.join(dir_data,'fmc_24_may_2019'))
    df = clean_fmc(df, quality = 'all same')
    df = df.groupby('site').apply(rmsd)
    df.hist()
    # print('[INFO] 10th percentile correlation = %0.2f'%df.quantile(1))
def rmse_vs_climatology():
    latlon = pd.read_csv(r"D:\Krishna\projects\vwc_from_radar\data\whittaker\nfmd_sites_climatology.csv")
    latlon.index = latlon.Site
    latlon['rmse'] = frame.groupby('site').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat'])))   
    latlon['R'] = frame.groupby('site').apply(lambda df: np.corrcoef(df['percent(t)'],df['percent(t)_hat'])[0,1]) 

    slope, intercept, r_value, p_value, std_err = stats.linregress(latlon['ppt'].values,latlon['rmse'].values)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (DC,2.3), sharey = True)
    
    # ax1.set_ylim(-1,1)
    sns.regplot(latlon.ppt, latlon.rmse, ax = ax1)
    ax1.set_xlabel('MAP (mm.yr$^{-1}$)')
    ax1.set_ylabel('Site RMSE')
    
    sns.regplot(latlon.temp, latlon.rmse, ax = ax2)
    ax2.set_xlabel('MAT ($^o$C)')
    ax2.set_ylabel('')
    
    sns.regplot(latlon.elevation, latlon.rmse, ax = ax3)
    #ax.scatter(latlon.elevation, latlon.rmse)
    ax3.set_xlabel('Elevation (m)')
    ax3.set_ylabel('')
    
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'rmse_climate_topo.jpg'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show()
# save_fig = True    
def climatology_maps():
    
    subplot_kw = dict(projection=ccrs.LambertConformal())
    states = os.path.join(dir_data,'usa_shapefile/west_usa/west_usa_shapefile_lcc.shp')
    shape_feature = ShapelyFeature(Reader(states).geometries(),
                                ccrs.LambertConformal(), edgecolor='black')
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(DC,SC), subplot_kw=subplot_kw)
    
    ax=ax1
    fname = os.path.join(dir_data,'whittaker/ppt_lcc.tif')
    ds = gdal.Open(fname)
    data = ds.ReadAsArray()
    data[data<0] = np.nan
    gt = ds.GetGeoTransform()
    extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
              gt[3] + ds.RasterYSize * gt[5], gt[3])
    my_cmap = sns.light_palette("royalblue", as_cmap=True)
    
    img = ax.imshow(data,extent = extent , 
                    origin='upper', transform=ccrs.LambertConformal(),cmap =  my_cmap,\
                    vmin = 0, vmax = 4000)
    ax.add_feature(shape_feature,facecolor = "None",linewidth=0.5)
    ax.outline_patch.set_edgecolor('white')
    # cax1.annotate('MAP (mm/yr)', xy = (0.,0.94), ha = 'left', va = 'bottom')
    fig.colorbar(img,ax=ax,fraction=0.036, pad=0.04, ticks = [0,1000,2000,3000,4000]) 
    ax.annotate('MAP (mm/yr)', xy=(0.8,1), xycoords='axes fraction',\
                ha='left',va='bottom')
    
    ###########################################################################
    ax=ax2
    fname = os.path.join(dir_data,'whittaker/temp_lcc.tif')
    ds = gdal.Open(fname)
    data = ds.ReadAsArray()
    data[data<-40] = np.nan
    gt = ds.GetGeoTransform()
    extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
              gt[3] + ds.RasterYSize * gt[5], gt[3])
    my_cmap = sns.light_palette("#feb308", as_cmap=True)
    
    
    img = ax.imshow(data,extent = extent , 
                    origin='upper', transform=ccrs.LambertConformal(),cmap =  my_cmap,\
                    vmin = 0, vmax = 25)
    ax.add_feature(shape_feature,facecolor = "None",linewidth=0.5)
    ax.outline_patch.set_edgecolor('white')    

    # cax1.annotate('MAP (mm/yr)', xy = (0.,0.94), ha = 'left', va = 'bottom')
    fig.colorbar(img,ax=ax,fraction=0.036, pad=0.04, ticks = [-10,0,10,20,25]) 
    ax.annotate('MAT ($^o$C)', xy=(0.8,1), xycoords='axes fraction',\
                ha='left',va='bottom')
    ###########################################################################
    ax=ax3
    fname = os.path.join(dir_data,'whittaker/elevation_lcc.tif')
    ds = gdal.Open(fname)
    data = ds.ReadAsArray().astype('float')
    data[data<-1000] = np.nan
    gt = ds.GetGeoTransform()
    extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
              gt[3] + ds.RasterYSize * gt[5], gt[3])
    my_cmap = sns.light_palette("seagreen", as_cmap=True)
    
    img = ax.imshow(data,extent = extent , 
                    origin='upper', transform=ccrs.LambertConformal(),cmap =  my_cmap,\
                    vmin = 0, vmax = 4000)

    # cax1.annotate('MAP (mm/yr)', xy = (0.,0.94), ha = 'left', va = 'bottom')
    fig.colorbar(img,ax=ax,fraction=0.036, pad=0.04, ticks = [0,1000,2000,3000,4000]) 
    ax.annotate('Elevation (m)', xy=(0.8,1), xycoords='axes fraction',\
                ha='left',va='bottom')
    ax.add_feature(shape_feature,facecolor = "None",linewidth=0.5)
    ax.outline_patch.set_edgecolor('white')
    
    ax1.annotate('a.', xy=(0.2, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')  
    ax2.annotate('b.', xy=(0.2, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')      
    ax3.annotate('c.', xy=(0.2, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')
    
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'climatology_map.jpg'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show()
def lc_bar():
    df = pd.DataFrame({'sites':frame.groupby('site').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat']))).sort_values()})
    df['Landcover'] = frame.groupby('site').apply(lambda df: df['forest_cover(t)'].astype(int).values[0])
    df['Landcover'] = encoder.inverse_transform(df['Landcover'].values)
    df = df.groupby('Landcover').count()
    # df.sites /= df.sites.sum()
    df['study_area'] = [0.055510608, 0.213934613, 0.039317878, 0.063920033, 0.42991891, 0.151684868]
    df/= df.sum()
    df.index = df.index.map(lc_dict)
    df['Landcover'] = df.index
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (DC,SC),sharey = True)
    
    colors = df.index.map(color_dict)
    sns.barplot(x="Landcover", y="sites", data=df, ax = ax1, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),\
                edgecolor = 'lightgrey',linewidth = 1,
                )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45,horizontalalignment='right')

    sns.barplot(x="Landcover", y="study_area", data=df, ax = ax2, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),\
                edgecolor = 'lightgrey',linewidth = 1,
                )
    ax2.legend_.remove()
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax1.set_ylabel('Proportion of sites')
    ax2.set_ylabel('Proportion of study area')
    ax2.set_xticklabels(ax1.get_xticklabels(), rotation=45,horizontalalignment='right')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc = 'upper right',prop={'size': 7},frameon = False)
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'lc_bar.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show() 
def site_cv():
    df = pd.DataFrame({'true_mean':frame.groupby('site').apply(lambda df: df['percent(t)'].mean()).sort_values()})
    df['true_sd'] = frame.groupby('site').apply(lambda df: df['percent(t)'].std())
    df['pred_mean'] = frame.groupby('site').apply(lambda df: df['percent(t)_hat'].mean())
    df['pred_sd'] = frame.groupby('site').apply(lambda df: df['percent(t)_hat'].std())
    df['pred_cv'] = df['pred_sd']/df['pred_mean']
    df['CV'] = df['true_sd']/df['true_mean']
    
    df['true_75'] = frame.groupby('site').apply(lambda df: df['percent(t)'].quantile(0.75))
    df['true_25'] = frame.groupby('site').apply(lambda df: df['percent(t)'].quantile(0.25))
    df['pred_75'] = frame.groupby('site').apply(lambda df: df['percent(t)_hat'].quantile(0.75))
    df['pred_25'] = frame.groupby('site').apply(lambda df: df['percent(t)_hat'].quantile(0.25))
    df['R'] = frame.groupby('site').apply(lambda df: np.corrcoef(df['percent(t)'],df['percent(t)_hat'])[0,1])
    df['N_obs'] = frame.groupby('site').apply(lambda df: len(df['percent(t)']))

    fig, _ = plt.subplots(1,2,figsize = (SC,0.5*SC))
    grid = plt.GridSpec(1,2, wspace=0.45, figure = fig)
    
    ax1 = plt.subplot(grid[0,0])
    ax2 = plt.subplot(grid[0, 1])

    plot_pred_actual(df['CV'], df['R'],ax = ax2,\
                 ms = 25, xlabel = "CV$_{obs}$", \
            ylabel = "$r$",mec = 'grey', mew = 0.3,
            annotation = False, oneone=False)
    ### fit
    l = loess(df['CV'],df['R'],span = 1,degree = 2,weights = df['N_obs'])
    l.fit()
    new_x = np.linspace(df['CV'].min(),df['CV'].max())
    pred = l.predict(new_x, stderror=True)
    conf = pred.confidence()
    lowess = pred.values
    ll = conf.lower
    ul = conf.upper
    pred = l.predict(df['CV'], stderror=False).values
    R2 = r2_score(df['R'],pred)    
    ax2.plot(new_x, lowess,color = 'grey',zorder = 1,alpha = 0.7)
    ax2.fill_between(new_x,ll,ul,color = 'grey',alpha=.33,zorder = 2, linewidth = 0)
    ## axis 
    # ax1.annotate('$R^2$=%0.2f'%R2, xy=(0.98, 0.2), xycoords='axes fraction',\
                # ha='right',va='top')
    ax2.set_ylim(-1,1.05)
    ax2.set_xlim(0,0.6)
    ax2.set_xticks([0,0.2,0.4,0.6])
    ax2.set_yticks([-1,-0.5,0,0.5,1])    
    ax2.set_aspect(0.3,'box')
    ax2.set_ylabel('$r$',labelpad = -1)
    plot_pred_actual(df['CV'],df['pred_cv'],ax = ax1,ticks = [0,0.2,0.4,0.6],\
        ms = 25, axis_lim = [0,0.6], xlabel = "CV$_{obs}$", \
        ylabel = "CV$_{est}$",mec = 'grey', mew = 0.3,annotation = False)
    R2= r2_score(df.CV, df.pred_cv,sample_weight =  df['N_obs'])
    print(R2)
    # ax2.annotate('$R^2$=%0.2f\nMBE=-0.1'%R2, xy=(0.1, 0.95), xycoords='axes fraction',\
                # ha='left',va='top')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    ax1.annotate('a.', xy=(-0.28, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')  
    ax2.annotate('b.', xy=(-0.28, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'scatter_plot_cv.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show()
save_fig = True    
def main():
    bar_chart()
    # landcover_table()
    # microwave_importance()
    # nfmd_sites()
    scatter_plot_all_3()
    # rmse_vs_climatology()
    # g=2
    # sites_QC()
    # climatology_maps()
    # lc_bar()    
    # site_mean_anomalies_fill()
    # site_cv()
if __name__ == '__main__':
    main()