# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 00:11:14 2017

@author: kkrao
"""
import os
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon, Patch

import cartopy.crs as ccrs
 


from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde

from osgeo import gdal

from dirs import dir_data, dir_codes, dir_figures
from fnn_smoothed_anomaly_all_sites import plot_importance, plot_usa
import pickle
from matplotlib.colors import ListedColormap

#mpl.rcParams.update({'figure.autolayout': True})
#


#%% Plot control settings
ZOOM=1.0
FS=10*ZOOM
PPT = 0
DPI = 300

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
              'Shrub/grassland' :'olive' ,
              'Shrubland':'darkgoldenrod',
              'Grassland':'lawngreen',
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
    
    
    ### sort by landcover then ascending
    lc_order = rmse.groupby('Landcover').RMSE.mean().sort_values()
    lc_order.loc[:] = range(len(lc_order))
    rmse['lc_order'] = rmse['Landcover'].map(lc_order)
    rmse.sort_values(by=['lc_order','RMSE'], inplace = True)
    rmse['Sites'] = range(len(rmse))
    ## SD
    rmse['SD'] = frame.groupby('site').apply(lambda df: df['percent(t)'].std())

    ### mean rmse values store
    lc_rmse = rmse.groupby('Landcover').RMSE.mean().sort_values()
    lc_rmse.name = 'lc_rmse'
    # lc_rmse.index = [6.5,55,72,95,106.5,122.5]
    rmse = pd.merge(rmse, lc_rmse, on = 'Landcover')
    # rmse.join(lc_rmse, on = 'Landcover')
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (DC,4))
    grid = plt.GridSpec(1,2, wspace=0.4, hspace=0.15, figure = fig)
           
    colors = lc_rmse.index.map(color_dict)
    sns.barplot(x="Sites", y="RMSE", data=rmse, ax = ax1, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),edgecolor = 'w',linewidth = 0.04,
                )
    ## plot mean rmse points
    ax1.plot(rmse.index, rmse.lc_rmse, '-',color='sienna',alpha = 0.8,\
             label = 'Overall landcover RMSE')
   
    # ax1.set_xticklabels(range(len(rmse)))
    ax1.set_xticks([])
    ax1.set_yticks([0,20,40,60,80])
    # ax1.set_xticklabels([0,25,50,75,100,124])
    change_width(ax1, 1)
    ax1.set_ylim(0,90)

 
    #name panels
    ax1.annotate('a.', xy=(-0.12, 1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')  
    ax2.annotate('b.', xy=(-0.12, 1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')      

    ####SD

    ### mean rmse values store
    lc_sd = frame.groupby('forest_cover(t)').apply(lambda df: df['percent(t)'].std())
    lc_sd.name = 'lc_sd'
    lc_sd.index = encoder.inverse_transform(lc_sd.index.astype(int))
    lc_sd.index = lc_sd.index.map(lc_dict)
    lc_sd.index.name = 'Landcover'

    # lc_rmse.index = [6.5,55,72,95,106.5,122.5]
    rmse = pd.merge(rmse, lc_sd, on = 'Landcover')    
    
    ##### plotting 
    colors = lc_rmse.index.map(color_dict)
    sns.barplot(x="Sites", y="SD", data=rmse, ax = ax2, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),edgecolor = 'w',linewidth = 0.04,
                )
    ## plot mean rmse points
    ax2.plot(rmse.index, rmse.lc_sd, '--',color='sienna',alpha = 0.8,\
             label = 'Overall landcover SD')
    ax2.set_ylim(0,90)
    ax2.set_xticks([])
    ax2.set_yticks([0,20,40,60,80])
    # ax1.set_xticklabels([0,25,50,75,100,124])
    change_width(ax2, 1)
    # ax2.set_ylim(-0.05,1.05)
    # ax2.set_ylabel('SD')
    h2,l2 = ax2.get_legend_handles_labels()
     # ax1.bar(rmse.index, rmse.values, width = 1)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(handles.pop(0))
    labels.append(labels.pop(0))
    handles.append(h2[0])
    labels.append(l2[0])
    
    ax1.legend(handles, labels,loc = 'upper left',prop={'size': 7})
    ax2.get_legend().remove()
    


    if save_fig:
        plt.savefig(os.path.join(dir_figures,'bar_plot.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show()


def bar_chart_sd():
    rmse = pd.DataFrame({'SD':frame.groupby('site').apply(lambda df: r2_score(df['percent(t)'],df['percent(t)_hat'])).sort_values()})
    rmse['Landcover'] = frame.groupby('site').apply(lambda df: df['forest_cover(t)'].astype(int).values[0])
    rmse['Landcover'] = encoder.inverse_transform(rmse['Landcover'].values)
    rmse['Landcover'] = rmse['Landcover'].map(lc_dict)
    
    
    ### sort by landcover then ascending
    lc_order = rmse.groupby('Landcover').SD.mean().sort_values()
    lc_order.loc[:] = range(len(lc_order))
    rmse['lc_order'] = rmse['Landcover'].map(lc_order)
    rmse.sort_values(by=['lc_order','SD'], inplace = True)
    rmse['Sites'] = range(len(rmse))
    ### mean rmse values store
    lc_rmse = rmse.groupby('Landcover').SD.mean().sort_values()
    lc_rmse.name = 'lc_rmse'
    # lc_rmse.index = [6.5,55,72,95,106.5,122.5]
    rmse = pd.merge(rmse, lc_rmse, on = 'Landcover')
    # rmse.join(lc_rmse, on = 'Landcover')
    
    fig, ax1 = plt.subplots(1,1, figsize = (SC,4))
    colors = lc_rmse.index.map(color_dict)
    # colors = ['darkslategrey','forestgreen','olive','darkgoldenrod','darkorange','lawngreen']
    sns.barplot(x="Sites", y="SD", data=rmse, ax = ax1, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),edgecolor = 'w',linewidth = 0.04,
                )
    ## plot mean rmse points
    # ax1.plot(rmse.index, rmse.lc_rmse, '-',color='sienna',alpha = 0.8,\
             # label = 'Mean landcover SD')
    # ax1.bar(rmse.index, rmse.values, width = 1)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(handles.pop(0))
    labels.append(labels.pop(0))
    
    ax1.legend(handles, labels,prop={'size': 7})
    ax1.set_ylabel('$R^2_{test}$')
    # ax1.set_xticklabels(range(len(rmse)))
    ax1.set_xticks([])
    # ax1.set_yticks([0,20,40,60,80])
    # ax1.set_xticklabels([0,25,50,75,100,124])
    change_width(ax1, 1)
    # ax1.set_ylim(0,90)
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'bar_plot_sd.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show()
def bar_chart_site_anomalies_R2():
    
    ndf = frame[['site','date','percent(t)','percent(t)_hat']]
    
    x = ndf.groupby(['site']).apply(lambda x: (x['percent(t)'] - x['percent(t)'].mean()))
    y = ndf.groupby(['site']).apply(lambda x: (x['percent(t)_hat'] - x['percent(t)_hat'].mean()))
    
    sa = pd.DataFrame({'true':x,'pred':y})
    sa.reset_index(inplace = True) 
    
    rmse = pd.DataFrame({'SD':sa.groupby('site').apply(lambda df: r2_score(df['true'],df['pred'])).sort_values()})
    rmse['Landcover'] = frame.groupby('site').apply(lambda df: df['forest_cover(t)'].astype(int).values[0])
    rmse['Landcover'] = encoder.inverse_transform(rmse['Landcover'].values)
    rmse['Landcover'] = rmse['Landcover'].map(lc_dict)
    
    
    ### sort by landcover then ascending
    lc_order = rmse.groupby('Landcover').SD.mean().sort_values()
    lc_order.loc[:] = range(len(lc_order))
    rmse['lc_order'] = rmse['Landcover'].map(lc_order)
    rmse.sort_values(by=['lc_order','SD'], inplace = True)
    rmse['Sites'] = range(len(rmse))
    ### mean rmse values store
    lc_rmse = rmse.groupby('Landcover').SD.mean().sort_values()
    lc_rmse.name = 'lc_rmse'
    # lc_rmse.index = [6.5,55,72,95,106.5,122.5]
    rmse = pd.merge(rmse, lc_rmse, on = 'Landcover')
    # rmse.join(lc_rmse, on = 'Landcover')
    
    fig, ax1 = plt.subplots(1,1, figsize = (SC,4))
    colors = lc_rmse.index.map(color_dict)
    # colors = ['darkslategrey','forestgreen','olive','darkgoldenrod','darkorange','lawngreen']
    sns.barplot(x="Sites", y="SD", data=rmse, ax = ax1, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),edgecolor = 'w',linewidth = 0.04,
                )
    ## plot mean rmse points
    # ax1.plot(rmse.index, rmse.lc_rmse, '-',color='sienna',alpha = 0.8,\
             # label = 'Mean landcover SD')
    # ax1.bar(rmse.index, rmse.values, width = 1)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(handles.pop(0))
    labels.append(labels.pop(0))
    
    ax1.legend(handles, labels,prop={'size': 7})
    ax1.set_ylabel('$R^2_{test}$')
    # ax1.set_xticklabels(range(len(rmse)))
    ax1.set_xticks([])
    ax1.set_ylabel('$R^2_{test}$(site-anomalies)')
    # ax1.set_yticks([0,20,40,60,80])
    # ax1.set_xticklabels([0,25,50,75,100,124])
    change_width(ax1, 1)
    # ax1.set_ylim(0,90)
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'bar_plot_sd.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show()
def bar_chart_time():
    rmse = pd.DataFrame({'SD':frame.groupby('site').apply(lambda df: r2_score(df['percent(t)'],df['percent(t)_hat'])).sort_values()})
    rmse['Landcover'] = frame.groupby('site').apply(lambda df: df['forest_cover(t)'].astype(int).values[0])
    rmse['Landcover'] = encoder.inverse_transform(rmse['Landcover'].values)
    rmse['Landcover'] = rmse['Landcover'].map(lc_dict)
    
    
    ### sort by landcover then ascending
    lc_order = rmse.groupby('Landcover').SD.mean().sort_values()
    lc_order.loc[:] = range(len(lc_order))
    rmse['lc_order'] = rmse['Landcover'].map(lc_order)
    rmse.sort_values(by=['lc_order','SD'], inplace = True)
    rmse['Sites'] = range(len(rmse))
    ### mean rmse values store
    rmse['$N_{obs}$'] = frame.groupby('site').apply(lambda df: df.shape[0])
    lc_rmse = rmse.groupby('Landcover').SD.mean().sort_values()
    lc_rmse.name = 'lc_rmse'
    # lc_rmse.index = [6.5,55,72,95,106.5,122.5]
    rmse = pd.merge(rmse, lc_rmse, on = 'Landcover')
    # rmse.join(lc_rmse, on = 'Landcover')
    
    fig, ax1 = plt.subplots(1,1, figsize = (SC,4))
    colors = lc_rmse.index.map(color_dict)
    # colors = ['darkslategrey','forestgreen','olive','darkgoldenrod','darkorange','lawngreen']
    sns.barplot(x="Sites", y="$N_{obs}$", data=rmse, ax = ax1, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),edgecolor = 'w',linewidth = 0.04,
                )
    ## plot mean rmse points
    # ax1.plot(rmse.index, rmse.lc_rmse, '-',color='sienna',alpha = 0.8,\
             # label = 'Mean landcover SD')
    # ax1.bar(rmse.index, rmse.values, width = 1)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(handles.pop(0))
    labels.append(labels.pop(0))
    
    ax1.legend(handles, labels,prop={'size': 7})
    # ax1.set_ylabel('$R^2_{test}$')
    # ax1.set_xticklabels(range(len(rmse)))
    ax1.set_xticks([])
    # ax1.set_yticks([0,20,40,60,80])
    # ax1.set_xticklabels([0,25,50,75,100,124])
    change_width(ax1, 1)
    # ax1.set_ylim(0,90)
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'bar_plot_sd.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show()

#%% performance by landcover table
def time_series():
   #%% timeseries for three sites
    new_frame = frame.copy()
    new_frame.index = pd.to_datetime(new_frame.date)
    rmse = new_frame.groupby('site').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat']))).sort_values()
    
    fig, (ax2, ax3, ax4) = plt.subplots(3,1,figsize = (SC,4))
    
    for site, ax in zip([0,63,-1], [ax2, ax3, ax4]):
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
        ax.legend(prop ={'size':7}, loc = 'lower right')
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
    
    ax2.annotate('a.', xy=(-0.2, 1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')  
    ax3.annotate('b.', xy=(-0.2, 1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold') 
    ax4.annotate('c.', xy=(-0.2, 1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold') 
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'time_series.eps'), \
                                 dpi =DPI, bbox_inches="tight")
    plt.show()
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
def plot_pred_actual(test_y, pred_y, cmap = ListedColormap(sns.cubehelix_palette().as_hex()), axis_lim = [-25,50],\
                 xlabel = "None", ylabel = "None", ticks = None,\
                 ms = 8, mec ='', mew = 0, ax = None):
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
    
    ax.plot(axis_lim,axis_lim, lw =.2, color = 'grey')
    
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yticks(ax.get_xticks())
#    ax.set_xticks([-50,0,50,100])
#    ax.set_yticks([-50,0,50,100])
    R2 = r2_score(x,y)
    model_rmse = np.sqrt(mean_squared_error(x,y))
    model_bias = np.mean(pred_y - test_y)
    ax.annotate('$R^2_{test}=%0.2f$\n$RMSE=%0.1f$\n$MBE=%0.1f$'%(np.floor(R2*100)/100, model_rmse, np.abs(model_bias)), \
                    xy=(0.03, 0.97), xycoords='axes fraction',\
                    ha='left',va='top')
    ax.set_xlim(axis_lim)
    ax.set_ylim(axis_lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # plt.tight_layout()
        
def scatter_plot():
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (DC,DC/3))
    grid = plt.GridSpec(1, 3, wspace=0.6, figure = fig)
    
    ax1 = plt.subplot(grid[0,0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[0, 2])
    
    
    x = frame['percent(t)'].values
    y = frame['percent(t)_hat'].values
    
    plot_pred_actual(x, y,ax = ax1,ticks = [0,100,200,300],\
        ms = 40, axis_lim = [0,300], xlabel = "$LFMC_{obs}$ (%)", \
        ylabel = "$LFMC_{est}$ (%)",mec = 'grey', mew = 0)
    

    t = frame.groupby('site')['percent(t)','percent(t)_hat'].mean()
    x = t['percent(t)'].values
    y = t['percent(t)_hat'].values
    
    plot_pred_actual(x, y,ax = ax2,\
                 ms = 40, axis_lim = [50,200], xlabel = "$\overline{LFMC_{obs}}$ (%)", \
            ylabel = "$\overline{LFMC_{est}}$ (%)",mec = 'grey', mew = 0.3, ticks = [50,100,150,200])
    
    
    
    ndf = frame[['site','date','percent(t)','percent(t)_hat']]
    
    x = ndf.groupby(['site']).apply(lambda x: (x['percent(t)'] - x['percent(t)'].mean())).values
    y = ndf.groupby(['site']).apply(lambda x: (x['percent(t)_hat'] - x['percent(t)_hat'].mean())).values
    
    plot_pred_actual(x, y,ax = ax3,ticks = [-100,-50,0,50,100],\
                 ms = 40, axis_lim = [-100,100], xlabel = "$LFMC_{obs} - \overline{LFMC_{obs}}$ (%)", \
            ylabel = "$LFMC_{est} - \overline{LFMC_{est}}$ (%)",mec = 'grey', mew = 0)
    ax1.annotate('a.', xy=(-0.28, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')  
    ax2.annotate('b.', xy=(-0.28, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')      
    ax3.annotate('c.', xy=(-0.28, 1.1), xycoords='axes fraction',\
                ha='right',va='bottom', weight = 'bold')
    if save_fig:
        plt.savefig(os.path.join(dir_figures,'scatter_plot.eps'), \
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
    

def seasonal_anomaly():
    seasonal_mean = pd.read_pickle(os.path.join(dir_data,'seasonal_mean_all_sites_%s_31_may_2019'%RESOLUTION))
    frame.date = pd.to_datetime(frame.date)
    frame['1M'] = frame.date.dt.month
    frame['SM'] = (2*frame['1M'] - 1*(frame.date.dt.day<=15)).astype(int)
    # <= because <15 is replaced with 15 in pandas SM
    frame['percent_seasonal_mean'] = np.nan
    for site in frame.site.unique():
        df_sub = frame.loc[frame.site==site,['site','date','SM','percent(t)']]
        if site not in seasonal_mean.columns:
            continue
        df_sub = df_sub.join(seasonal_mean.loc[:,site].rename('percent_seasonal_mean'), on = RESOLUTION)
        frame.update(df_sub)
        # frame.loc[frame.site==site,['site','date','mod','percent(t)','percent_seasonal_mean']]
    newframe = frame.dropna(subset = ['percent_seasonal_mean'])
    
    ### manually calculating because if there is a site with only 1 observation per MoY, its rmsd will be zero
    #there are 97 such sites

    x = newframe['percent(t)'] - newframe['percent_seasonal_mean'].values
    y = newframe['percent(t)_hat'] - newframe['percent_seasonal_mean'].values
    print('[INFO] Seasonal anomaly RMSE : %.1f' %mean_squared_error(x,y)**0.5)
    print('[INFO] Seasonal anomaly R2 : %.2f' %r2_score(x,y))

save_fig = False

def main():
    # seasonal_anomaly()
    # bar_chart()
    # time_series()
    # bar_chart_sd()
    # bar_chart_time()
    bar_chart_site_anomalies_R2()
    # scatter_plot()
    # landcover_table()
    # microwave_importance()
    # nfmd_sites()
    
if __name__ == '__main__':
    main()