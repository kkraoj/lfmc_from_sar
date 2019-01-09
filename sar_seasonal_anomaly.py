1# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 02:11:42 2018

@author: kkrao
"""

import os
import pandas as pd 
from dirs import dir_data, dir_figures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import neighbors
from sklearn.svm import SVR
from scipy.signal import savgol_filter
## general inputs
sns.set(style='ticks')
os.chdir(dir_data)
knn = neighbors.KNeighborsRegressor(n_neighbors=10, weights = "uniform", p = 2)

site_type =  pd.read_excel("fuel_moisture/seasonality_of_locations.xlsx", index_col = 0, usecols = 2)
window = '30d'

def seasonal_anomaly(Df, variable, save = False):
    """
    index should be datetime
    
    """
    Df['doy'] = Df.date.dt.dayofyear
    sa = pd.DataFrame()
    for site in Df.site.unique():
        df = Df.loc[Df.site == site,:]
        if len(df)<10:
            continue
        new_index = df.groupby(df.date).mean().resample('1d').asfreq().index
        mean = knn.fit(df['doy'].values.reshape(-1,1), df[variable]).predict(np.arange(1, 367).reshape(-1, 1))
        mean = pd.Series(mean[new_index.dayofyear-1], index = new_index).rolling(window = '10d').mean()       
        absolute = knn.fit(df.date.values.reshape(-1,1), df[variable]).predict(new_index.values.reshape(-1, 1))
        absolute = pd.Series(index = new_index, data = absolute).rolling(window = '10d').mean()       
        anomaly = absolute - mean
        anomaly.name = site
        sa = pd.concat([sa,anomaly], axis = 1)
    sa.name = variable+'_anomaly'
    if save:
        sa.to_pickle(os.path.join(dir_data,sa.name))
    return sa
###############################################################################
#######compute anomalies
#pass_type = 'am'
#obs = pd.read_pickle("sar_%s_500m_angle_corr"%pass_type)
##obs.loc[obs.vh<=-30,'vh'] = np.nan ###not required for angle corrected vh
##obs.loc[obs.vh>0,'vh'] = np.nan ###not required for angle corrected vh
##obs.loc[obs.vv<=-20,'vv'] = np.nan ###not required for angle corrected vh
##obs.loc[obs.vv>0,'vv'] = np.nan ###not required for angle corrected vh
#obs.dropna(subset = ['vh','vv'], how = 'any', inplace = True)
#vh_a = seasonal_anomaly(obs,'vh')
#vv_a = seasonal_anomaly(obs,'vv')
#vh_a.to_pickle('vh_%s_anomaly'%pass_type)
#vv_a.to_pickle('vv_%s_anomaly'%pass_type)

#meas = pd.read_pickle('vwc')
#fm_a = seasonal_anomaly(meas,'percent')
#fm_a.to_pickle('fm_anomaly')
    
#df = pd.read_pickle("opt_500m")
#df.rename(columns = {'b2':'blue', 'b3':'green', 'b4':'red', 'b8':'nir', 'b11':'swir'}, inplace = True)
#df['ndvi'] = (df.nir - df.red)/(df.nir + df.red)
#df['ndwi'] = (df.nir - df.swir)/(df.nir + df.swir)
#df.dropna(subset = [ 'blue', 'green', 'red', 'nir', 'ndvi','ndwi'],how = 'any',inplace = True)
#df.loc[:,[ 'blue', 'green', 'red', 'nir']]/=1e4
##for var in [ 'blue', 'green', 'red', 'nir']:
##    df_a =  seasonal_anomaly(df,var, save = True)
#df_a =  seasonal_anomaly(df,'ndvi', save = True)
#df_a =  seasonal_anomaly(df,'ndwi', save = True)
###############################################################################
### compute correlations between anomalies
#pass_type = 'pm'
#vh_a = pd.read_pickle('vh_%s_anomaly'%pass_type)
#fm_a = pd.read_pickle('fm_anomaly')
#corrs=[]
#for site in vh_a.columns:
#    if site in fm_a.columns:
#        corr = vh_a[site].corr(fm_a[site])
#        if corr>0.5:
#            corrs.append(corr)

################################################################################
### plotting ts
#pass_type = 'pm'
#obs = pd.read_pickle("sar_%s_500m"%pass_type)
#obs.index = obs.date
#obs.loc[obs.vh<=-30,'vh'] = np.nan
#obs.loc[obs.vh>0,'vh'] = np.nan
#obs.loc[obs.vv<=-20,'vv'] = np.nan
#obs.loc[obs.vv>0,'vv'] = np.nan
#meas = pd.read_pickle('vwc')
#meas.index = meas.date
#meas = meas.loc[meas.date>='2015-01-01',:]
#meas = meas.loc[meas.percent>=20,:]
#opt = pd.read_pickle("opt_500m")
#opt.index = opt.date
#opt = opt.loc[opt.qa60==0,:]
#opt.b3/=1e4
#opt.dropna(subset = ["b3"], inplace = True)
#opt.interpolate(method = "linear",inplace = True)
#
#c1 = 'y'
#c2 = 'maroon'
#ctr = 0
#var = "vv"
#sites=[]
#plot_ = True
#for site in obs.site.unique():
##for site in ['Muskrat']:
#    obs_sub = obs.loc[obs.site==site,:].copy()
#    if len(obs_sub.loc[obs_sub.index>='2017-01-01',:])<20  or site not in meas.site.values:
#        continue
#    elif site in meas.site.values:
#        meas_sub  = meas.loc[meas.site==site,:]
#        if len(meas_sub.fuel.unique())>1:
#            ##### select fuel with most readings
#            meas_sub = meas_sub.loc[meas_sub.fuel==meas_sub.fuel.value_counts().idxmax(),:]
#        if len(meas_sub.index.year.unique())<=1\
#            or ~(meas_sub.groupby(meas_sub.index.year).apply(lambda df: len(df))>=10).all():
##            or len(meas_sub.fuel.unique())>1:
##            or ~(meas_sub.groupby(meas_sub.index.month).apply(lambda df: len(df))>=10).all():
#            continue
#        
#            
#        
##            or ~(meas_sub.groupby(meas_sub.index.year).apply(lambda df: len(df))>=10).all():
##            continue
##            
##            or (meas_sub.loc[meas_sub.index>="2015-01-01",:].date.diff().abs()>'180days').any()            
#    if plot_:     
#        obs_sub.dropna(subset = [var], inplace = True)
#        x_ = obs_sub.groupby(obs_sub.index).mean().resample('1d').asfreq().index.values
#        y_ = obs_sub.groupby(obs_sub.index).mean().resample('1d').asfreq()[var].interpolate().rolling(window = window).mean()
#    #    y_ = knn.fit(obs_sub.index.values.reshape(-1,1), obs_sub[var].dropna()).predict(x_.reshape(-1, 1))
#        x2_ = meas_sub.groupby(meas_sub.index).mean().resample('1d').asfreq().index.values
#        
##        y2_ = knn.fit(meas_sub.index.values.reshape(-1,1), meas_sub.percent.dropna()).predict(x2_.reshape(-1, 1))    
#        y2_ = meas_sub.groupby(meas_sub.index).mean().resample('1d').asfreq().percent.interpolate().rolling(window = window).mean()
#        opt_sub = opt.loc[opt.site==site,:]
#        opt_sub = opt_sub.loc[opt_sub.b3<=opt_sub.b3.quantile(0.95),:]
#        x3_ = opt_sub.groupby(opt_sub.index).mean().resample('1d').asfreq().index.values
#        y3_ = opt_sub.groupby(opt_sub.index).mean().resample('1d').asfreq()['b3'].interpolate().rolling(window = window).mean()
#    #    y3_ = knn.fit(opt_sub.index.values.reshape(-1,1), opt_sub.b3.dropna()).predict(x3_.reshape(-1, 1))    
#        corr_g = pd.Series(y_,x_).corr(pd.Series(y3_,x3_))
#    #    print(corr_g)    
#  
#        fig, ax = plt.subplots(figsize = (4,1.5))
#        obs_sub[var].plot(ax = ax, marker = 'o', ms = 3, ls='none', c= 'y',mfc = "none",mew = 0.2,  mec = "y", label = 'loess')
#    #    obs_sub[var].resample('1d').asfreq().interpolate('linear').plot(ax = ax, marker = 'None', ms = 3, ls='-', c= 'y',mfc = "none",mew = 0.5,  mec = "y", label = 'loess')
#        ax.plot(x_,y_, color = 'y', lw = 1)
#        ax.set_ylabel('$\sigma_{VH}$(dB)', color = 'y')
#        ax.set_title(site)
#        ax.tick_params('y', colors='y')
#        
#        ax2 = ax.twinx()
#        ax2.tick_params('y', colors='maroon')
#    ##    ax2.set_ylabel('FM (%)', color = 'maroon')
#        meas_sub.percent.sort_values().astype(np.int).plot(ax = ax2, marker = 'o', ms = 3, ls='none', mfc = "none",mew = 0.2,  c = "maroon",mec = "maroon", label = 'fm', lw = 1.5)
#        ax2.plot(x2_,y2_, color = 'maroon', lw = 1)
#    #    ax2.plot(x2_,y2_, color = c2, lw = 1)
#        fm_monthly = y2_.groupby(y2_.index.month).mean()
#        corr = pd.Series(y_,x_).corr(pd.Series(y2_,x2_))
#        ax.annotate('$R$ = %0.2f'%corr, xy=(1, 0.84),ha = "right", xycoords='axes fraction', color = 'y')  
#        
#        ax3 = ax.twinx()
#        
#        opt_sub.b3.plot(ax=ax3,  marker = 'o', ms = 2, ls='none', mfc = "none",mew = 0.2,  mec = "g", label = 'loess')
#        ax3.plot(x3_,y3_, color = 'g', lw = 1)
#        ax3.set_ylabel('Green', color='g')
#        ax3.tick_params('y', colors='g')
#        ax3.spines["right"].set_position(("axes", 1.15))
#    
#        ax.set_xlabel("")
#        ax.set_xlim(xmin = pd.to_datetime('2015-10-01'))
#        ax.annotate(r'FM (%)', color = 'maroon', xy=(1.0, 1.07), xycoords='axes fraction')
#        plt.show()
#    ctr+=1
#    sites.append(site)
#print('Total sites = %s'%ctr)
###############################################################################
### plotting anomalies as scatter plot
#pass_type = 'am'
#var = "vh"
#vh_a = pd.read_pickle('%s_%s_anomaly'%(var,pass_type))
#fm_a = pd.read_pickle('fm_anomaly')
#corrs=[]
#ctr = 0
#for site in sites:
#    if site in fm_a.columns:
#        x,y= vh_a[site].align(fm_a[site], join = 'left')
#        if x.corr(y) > -1:
#            fig, ax = plt.subplots(figsize= (1.5,1.5))
#            ax.scatter(x,y,facecolor="none",edgecolor = "b", s= 10)
#            ax.annotate('$R$ = %0.2f'%x.corr(y), xy=(0.02, 0.84),ha = "left", xycoords='axes fraction', weight = 'bold', color = 'peru')  
#            ax.set_xlabel('$\sigma_{VH}$ anomaly (dB)')
#            ax.set_ylabel('FM anomaly (% FM)')
#            ax.set_title(site)
#            plt.show()
#            ctr+=1
#print('Total sites = %s'%ctr)
###############################################################################
### plotting anomaly derivation
zoom = 2
sns.set(font_scale=zoom, style = 'ticks')

site = "Smith Ranch"
variable = "percent"
df = pd.read_pickle('vwc')
df = df.loc[df.site==site,:]
df = df.loc[df.date>='2015-01-01',:]
df['doy'] = df.date.dt.dayofyear
new_index = df.groupby(df.date).mean().resample('1d').asfreq().index
mean = knn.fit(df['doy'].values.reshape(-1,1), df[variable]).predict(np.arange(1, 367).reshape(-1, 1))
mean = pd.Series(mean[new_index.dayofyear-1], index = new_index)        
absolute = knn.fit(df.date.values.reshape(-1,1), df[variable]).predict(new_index.values.reshape(-1, 1))
absolute = pd.Series(index = new_index, data = absolute)
anomaly = absolute - mean
anomaly.name = site
mean = mean.rolling(window = '30d').mean()
absolute = absolute.rolling(window = '30d').mean()
fig, ax = plt.subplots(figsize= (zoom*3,zoom*1.5))
mean.plot(ax = ax, label = 'mean', color = 'k', lw = zoom, ls = '--')
absolute.plot(ax = ax, label = 'raw', color = 'k', lw = zoom)
ax.fill_between(mean.index, mean.values, absolute.values, where=absolute.values <= mean.values,\
                color = 'tomato', label = '-anomaly')
ax.fill_between(mean.index, mean.values, absolute.values, where=absolute.values> mean.values,\
                color = 'dodgerblue', label = '+anomaly')
ax.set_ylabel('FMC (%)')
ax.set_xlabel("")
ax.legend(bbox_to_anchor =(1,1))
plt.savefig(os.path.join(dir_figures,'anomaly_derivation.tiff'), dpi = 600,\
                    bbox_inches="tight")


###############################################################################
### plotting raw scatter plots
def mark_gaps(df, max_delta = '60days'):
    df['delta'] = df['date'].diff()
    new_dates = df.loc[df.delta.abs()>=max_delta,'delta']
    new_dates = (new_dates.index-new_dates/2)
    for date in new_dates:
        df.loc[date,:] = np.nan
    df.sort_index(inplace = True)
    return df

pure_species_sites = []
#for site in ['Ponderosa Basin Ceanothus, Buckbrush New', 'Throckmorton',
#       'Ponderosa Basin Ceanothus, Buckbrush Old',
#       'Ponderosa Basin Manzanita, Whiteleaf New',
#       'Ponderosa Basin Manzanita, Whiteleaf Old']:
#    if len(meas.loc[meas.site==site,'fuel'].unique())==1 and \
#        len(meas.loc[(meas.site==site)&(meas.date>='2015-01-01'),:])>30 and\
#        site in obs.site.values:
#        if len(obs.loc[obs.site==site,:])<20:
#            continue
#        pure_species_sites.append(site)
#svr_rbf = SVR(kernel='rbf', C=5, epsilon = .5)
#
#ctr=0
#for site in ['Tapo Canyon, Simi Valley']:
#    fig, ax = plt.subplots(figsize = (4,1.5))
#    obs_sub = obs.loc[obs.site==site,'vh'].dropna().copy()
##    obs_sub.drop(obs_sub.loc[obs_sub.diff().abs()>=2].index, inplace = True)
#    obs_sub.plot(ax = ax, marker = 'o', ms = 3, ls='-', mfc = "none",mew = 0.5,  c = "y",mec = "y", label = 'vh',lw = 1)
#    ax2 = ax.twinx()
#    meas_sub = meas.loc[meas.site==site,:].copy()
#    meas_sub = mark_gaps(meas_sub)
#    meas_sub.percent.plot(ax = ax2, marker = 'o', ms = 3, ls='-', mfc = "none",mew = 0.5,  c = "maroon",mec = "maroon", label = 'vh')
##    #######SVR
##    x_ = obs.loc[obs.site==site,:].index.sort_values().values.reshape(-1,1)
##    y_rbf = svr_rbf.fit(x_, obs.loc[obs.site==site,'vh']).predict(x_)
##    ax.plot(x_, y_rbf, ms = 3, ls='-', mfc = "none",mew = 0.5,  c = "y",mec = "y", label = 'vh',lw = 1)    
#
#    ax.set_ylabel('$\sigma_{VH}$(dB)', color = 'y')
#    ax.tick_params('y', colors='y')
#    ax2.set_ylabel('FM (%)', color = 'maroon')
#    ax2.tick_params('y', colors='maroon')
#    ax.set_title(site)
#    plt.show()
#    ctr+=1
#print('Total sites = %s'%ctr)
################################################################################
#########anomaly TS
#vh_a = pd.read_pickle('vh_am_anomaly')
#fm_a = pd.read_pickle('fm_anomaly')
#fm_a = fm_a.loc[(fm_a.index.year>=2015)&(fm_a.index<='2018-06-01'),:]
#for site in pure_species_sites:
#    if site not in vh_a.columns:
#        continue
#    fig, ax = plt.subplots(figsize = (4,1.5))
#    vh_a.loc[:,site].plot(ax = ax, ms = 3, ls='-', mfc = "none",mew = 0.5,  c = "y",mec = "y", label = 'vh',lw = 1)
#    ax2 = ax.twinx()
#    fm_a.loc[:,site].plot(ax = ax2, ms = 3, ls='-', mfc = "none",mew = 0.5,  c = "maroon",mec = "maroon", label = 'vh')
#
#    ax.set_ylabel('$\sigma_{VH}$ anomaly(dB)', color = 'y')
#    ax.tick_params('y', colors='y')
#    ax2.set_ylabel('FM anomaly(%)', color = 'maroon')
#    ax2.tick_params('y', colors='maroon')
#    ax.set_title(site)
#    plt.show()

    