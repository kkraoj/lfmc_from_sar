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
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
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
            50: 'closed broadleaf deciduous',
            70: 'closed needleleaf evergreen',
            90: 'mixed forest',
            100:'mixed forest',
            110:'shrub/grassland',
            120:'grassland/shrubland',
            130:'closed to open shrub',
            140:'grass',
            150:'sparse vegetation',
            160:'regularly flooded forest'}

pkl_file = open(os.path.join(dir_data,'encoder.pkl'), 'rb')
encoder = pickle.load(pkl_file) 
pkl_file.close()

#%% RMSE vs sites. bar chart

frame = pd.read_csv(os.path.join(dir_data,'model_predictions_all_sites.csv'))



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
    ### mean rmse values store
    lc_rmse = rmse.groupby('Landcover').RMSE.mean().sort_values()
    lc_rmse.index = [6.5,55,72,95,106.5,122.5]
    
    fig, _ = plt.subplots(2,3, figsize = (DC,4))
    grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.15, figure = fig)
    ax1 = plt.subplot(grid[0:, 0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, 1])
    ax4 = plt.subplot(grid[2, 1])
    
    def change_width(ax, new_value) :
        for patch in ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_value
    
            # we change the bar width
            patch.set_width(new_value)
    
            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)
    color_dict = {'closed broadleaf deciduous':'darkorange',
                  'closed needleleaf evergreen': 'forestgreen',
                  'mixed forest':'darkslategrey',
                  'shrub/grassland' :'olive' ,
                  'closed to open shrub':'darkgoldenrod',
                  'grass':'lawngreen',
                  }         

    colors = ['darkslategrey','forestgreen','olive','darkgoldenrod','darkorange','lawngreen']
    sns.barplot(x="Sites", y="RMSE", data=rmse, ax = ax1, hue = 'Landcover', \
                dodge = False,palette=sns.color_palette(colors),edgecolor = 'w',linewidth = 0.1,
                )
    ## plot mean rmse points
    ax1.plot(lc_rmse,'s',color='lightgrey',ms = 3.5, mec = 'slategrey',mew = 0.4,\
             label = 'mean landcover rmse')
    # ax1.bar(rmse.index, rmse.values, width = 1)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(handles.pop(0))
    labels.append(labels.pop(0))
    
    ax1.legend(handles, labels,loc = 'upper left',prop={'size': 7})
    
    # ax1.set_xticklabels(range(len(rmse)))
    ax1.set_xticks([0,25,50,75,100,124])
    ax1.set_xticklabels([0,25,50,75,100,124])
    change_width(ax1, 1)

    #%% timeseries for three sites
    new_frame = frame.copy()
    new_frame.index = pd.to_datetime(new_frame.date)
    rmse = new_frame.groupby('site').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat']))).sort_values()
    
    for site, ax in zip([0,63,-1], [ax2, ax3, ax4]):
        sub = new_frame.loc[new_frame.site == rmse.index[site]]
        
        lc = lc_dict[encoder.inverse_transform(sub['forest_cover(t)'].\
                                                   astype(int).unique()[0])]
        if ax == ax4:       
            sub.plot(y = 'percent(t)', linestyle = '-', markeredgecolor = 'k', ax = ax,\
                marker = 'o', label = 'actual', color = 'grey', mew =0.3,ms = 3,linewidth = 1 )
        else:
            sub.plot(y = 'percent(t)', linestyle = '-', markeredgecolor = 'k', ax = ax,\
                marker = 'o', label = '_nolegend_', color = 'grey', mew =0.3,ms = 3,linewidth = 1 )
        
        sub.plot(y = 'percent(t)_hat', linestyle = '--', markeredgecolor = 'k', ax = ax,\
                marker = 'o', label = 'predicted',color = color_dict[lc], mew= 0.3, ms = 3, lw = 1, rot = 0)
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

#%% performance by landcover table
def landcover_table():
    table = pd.DataFrame({'RMSE':frame.groupby('forest_cover(t)').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat'])))}).round(1)
    table['R2'] = frame.groupby('forest_cover(t)').apply(lambda df: np.corrcoef(df['percent(t)'],df['percent(t)_hat'])[0,1]**2).round(2)
    table['N_obs'] = frame.groupby('forest_cover(t)').apply(lambda df: df.shape[0])
    table['N_sites'] = frame.groupby('forest_cover(t)').apply(lambda df: len(df.site.unique()))
    table['Bias'] = frame.groupby('forest_cover(t)').apply(lambda df: (df['percent(t)'] - df['percent(t)_hat']).mean()).round(1)
    ### works only with original encoder!!
    
    table.index = encoder.inverse_transform(table.index.astype(int))
    table.index = table.index.map(lc_dict)
    table.index.name = 'landcover'
    table = table[['N_sites','N_obs','RMSE','R2','Bias']]
    
    print(table)
    # table.to_excel('model_performance_by_lc.xls')
    # table.to_latex('model_performance_by_lc.tex')

#%% prediction scatter plots after CV

def scatter_plot():
    
    def plot_pred_actual(test_y, pred_y, cmap = ListedColormap(sns.cubehelix_palette().as_hex()), axis_lim = [-25,50],\
                     xlabel = "None", ylabel = "None", ticks = None,\
                     ms = 8, mec ='', mew = 0, ax = None):
        # plt.axis('scaled')
        ax.set_aspect('equal', 'box')

        x = test_y
        y = pred_y

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
        ax.annotate('$R^2=%0.2f$\n$RMSE=%0.1f$\n$Bias=%0.1f$'%(np.floor(R2*100)/100, model_rmse, model_bias), \
                        xy=(0.03, 0.97), xycoords='axes fraction',\
                        ha='left',va='top')
        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        # plt.tight_layout()

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (DC,DC/3))
    grid = plt.GridSpec(1, 3, wspace=0.6, figure = fig)
    
    ax1 = plt.subplot(grid[0,0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[0, 2])
    
    
    x = frame['percent(t)'].values
    y = frame['percent(t)_hat'].values
    
    plot_pred_actual(x, y,ax = ax1,ticks = [0,100,200,300],\
        ms = 40, axis_lim = [0,300], xlabel = "LFMC (%)", \
        ylabel = "$\hat{LFMC}$ (%)",mec = 'grey', mew = 0)
    

    t = frame.groupby('site')['percent(t)','percent(t)_hat'].mean()
    x = t['percent(t)'].values
    y = t['percent(t)_hat'].values
    
    plot_pred_actual(x, y,ax = ax2,\
                 ms = 40, axis_lim = [50,200], xlabel = "$\overline{LFMC_s}$ (%)", \
            ylabel = "$\hat{\overline{LFMC_s}}$ (%)",mec = 'grey', mew = 0.3, ticks = [50,100,150,200])
    
    
    
    ndf = frame[['site','date','percent(t)','percent(t)_hat']]
    ndf.dropna(inplace = True)
    
    x = ndf.groupby(['site']).apply(lambda x: (x['percent(t)'] - x['percent(t)'].mean())).values
    y = ndf.groupby(['site']).apply(lambda x: (x['percent(t)_hat'] - x['percent(t)'].mean())).values
    
    plot_pred_actual(x, y,ax = ax3,ticks = [-100,-50,0,50,100],\
                 ms = 40, axis_lim = [-100,100], xlabel = "LFMC - $\overline{LFMC_s}$ (%)", \
            ylabel = "$\hat{LFMC} - \hat{\overline{LFMC_s}}$ (%)",mec = 'grey', mew = 0)
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
    
    
save_fig = True

def main():
    bar_chart()
    # scatter_plot()
    # landcover_table()
    # microwave_importance()
    
if __name__ == '__main__':
    main()