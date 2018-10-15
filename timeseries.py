# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 23:20:17 2018

@author: kkrao
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from plots_sar_vwc import clean_xy
from mpl_toolkits.axes_grid1 import make_axes_locatable
def truncate_colormap(cmap = plt.cm.viridis, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

pd.set_option('display.max_columns', 30)

os.chdir("D:/Krishna/projects/vwc_from_radar/data")

Site = "17Rd"
start_date = "2015-04-01"
meas = pd.read_pickle("df_vwc")
meas.index = meas.meas_date
end_date = meas.meas_date.max()
#meas = meas.loc[meas.meas_date>=start_date,:]

obs = pd.read_pickle("df_sar_pm")
obs.index = obs.obs_date
obs['vh_angle_corr'] = obs.VH*np.cos(np.deg2rad(obs.angle))
obs['vv_angle_corr'] = obs.VV*np.cos(np.deg2rad(obs.angle))

#obs = obs.loc[obs.obs_date<=end_date,:]


opt = pd.read_pickle("df_optical")
opt.index = opt.obs_date
opt = opt.loc[opt.obs_date<=end_date,:]
opt.B3/=1e4
opt.loc[opt.B3<=0.2,"B3"] = np.nan
opt.dropna(subset = ["B3"], inplace = True)
#opt.interpolate(method = "linear",inplace = True)

counter = 0

both = pd.read_pickle('df_all')
def ignore_multi_spec_fun(df, pct_thresh = 20): 
    to_select = pd.DataFrame()
    for site in df.Site.unique():
        d = df.loc[df.Site==site,:]
        for date in d.meas_date.unique():
            x = d.loc[d.meas_date == date,:]
            if x.Percent.max() - x.Percent.min()<= pct_thresh:
                to_select = to_select.append(x)
    return to_select


### TS of opt and meas
#for Site in meas.Site.unique():
#    if not(Site in obs.Site.values):
##        print(Site)
#        continue
#    sub = meas.loc[meas.Site==Site,["meas_date","Percent", "Fuel"]]
#    if len(sub.Fuel.unique())>1 or len(sub)<=5:
#        continue
#    counter+=1
#    sub.dropna(inplace =True)
#    fig, ax = plt.subplots(figsize = (4,3) )
#    plt.xticks(rotation=70)
#    
#    ax.scatter(x = sub.meas_date.values,y =sub.Percent.values, color = "orange", edgecolor = 'k')
#    ax.set_ylabel('Fuel Moisture (%)', color='orange')
#    ax.tick_params('y', colors='orange')
#    
#    ax2 = ax.twinx()
#    obs.loc[obs.Site==Site,"vh_angle_corr"].plot(ax=ax2, color = "violet")
##    ax2.set_ylabel(r'$\sigma_{VH}\ (dB)$', color = 'violet')
#    ax2.tick_params('y', colors='violet')
#    
#    ax3 = ax.twinx()
#    opt.loc[opt.Site==Site,"B3"].plot(ax=ax3, color = "g", rot = 90)
##    ax3.set_ylabel('Green', color='g')
#    ax3.tick_params('y', colors='g')
#    ax3.spines["right"].set_position(("axes", 1.15))
#
#    ax.set_xlim([start_date, end_date])
#    ax.set_xlabel("")
#    ax.annotate(r'$\sigma_{VH}(dB)$', color = 'violet', xy=(0.95, 1.05), xycoords='axes fraction')
#    ax.annotate(r'Green', color = 'g', xy=(1.13, 1.05), xycoords='axes fraction')
#    
#    
#    plt.show()
##print(counter)

#### FM TS
def norm(Df, by = "Site"):
    df = Df.copy()
    df.index = df.Site
    df.Percent = df.groupby(by).transform(lambda x: (x - x.mean())/x.std())
#    df.Percent = df.Percent - df.groupby(by).Percent.mean()
    return df
    
#dir_data = "D:/Krishna/projects/vwc_from_radar/data/fuel_moisture"
#os.chdir(dir_data)
#files = os.listdir(dir_data+'/raw/')
#Df = pd.DataFrame()
#for file in files:
#    Df = pd.concat([Df, pd.read_table('raw/'+file)])
#Df.drop("Unnamed: 7", axis = 1, inplace = True)
#Df["meas_date"] = pd.to_datetime(Df["Date"])
#meas = Df.copy()   

#
#high_ndvi_sites = both.groupby("Site").apply(lambda df:(df.ndvi>=0.1).all())
#high_ndvi_sites = high_ndvi_sites.loc[high_ndvi_sites==True]
#
#filter =\
#    (~meas.Fuel.isin(['10-Hour', '100-Hour', '1000-Hour']))\
#    &(meas.Percent<=200)\
#    &(meas.Percent>=40)\
#    &meas.Site.isin(high_ndvi_sites.index)
#meas = meas.loc[filter,:]
#meas = ignore_multi_spec_fun(meas, pct_thresh =10)
#meas = norm(meas)
#fig, ax = plt.subplots(figsize = (3,3))
#x,y,z = clean_xy(meas.meas_date.dt.dayofyear.values, meas.Percent.values)
#ax.scatter(x,y, c = z,cmap = "viridis")
#ax.set_ylabel("Fuel moisture (%)")
#ax.set_ylabel(r"$\frac{FM - \mu_{site}(FM)}{ \sigma_{site}(FM)}$")
#ax.set_xlabel("Day of Year")
#ax.set_xticks(np.linspace(0,365,5))
#ax.set_xticklabels(["Jan","Apr","Jul","Oct"])
##ax.set_ylim([-2,4])


### TS of meas by DoY
meas = pd.read_pickle("df_vwc_historic")

filter =\
    (~meas.fuel.isin(['10-Hour', '100-Hour', '1000-Hour']))\
    &(meas.percent<=200)\
    &(meas.percent>=40)
meas = meas.loc[filter,:]
counter = 0
degree = 3
#for site in ['Dubois']:
#
#    sub = meas.loc[meas.site==site,["meas_date","percent", "fuel"]]
#    if len(sub.fuel.unique())>1 or len(sub)<=25 or \
#    len(sub.meas_date.dt.year.unique()) < 3 or site not in obs.Site.values:
#        continue
#    counter+=1
#    sub.dropna(inplace =True)
#    sub.sort_values(by ='meas_date', inplace = True)
##    print(site)
#    fig, ax = plt.subplots(figsize = (3,3))
#    start_year, end_year  = sub.meas_date.dt.year.min(), sub.meas_date.dt.year.max()
#    years = np.arange(start_year, end_year+1)
#    bounds = np.append((np.sort(sub.meas_date.dt.year.unique())),end_year+1)
#    
#    cmap = plt.cm.viridis
#    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#    
#    plot = ax.scatter(sub.meas_date.dt.dayofyear.values,sub.percent.values, \
#                      c = sub.meas_date.dt.year.values,\
#                      cmap =  cmap, label = 'Year', norm = norm, edgecolor = 'grey', linewidth=1)
#    ## FM fit
#    meas_fit= np.poly1d(np.polyfit(sub.meas_date.dt.dayofyear.values, sub.percent.values, degree))
#    xd = range(sub.meas_date.dt.dayofyear.min(), sub.meas_date.dt.dayofyear.max())
#    ax.plot(xd, meas_fit(xd), ls = '--', color = 'k', label = 'FM-fit')
#    ax.set_ylabel("Fuel moisture (%)")
##    ax.set_ylabel(r"$\frac{FM - \mu_{site}(FM)}{ \sigma_{site}(FM)}$")
#    ax.set_xlabel("Day of Year")
#    ax.set_xlim(0,365)
#    ax.set_xticks(np.linspace(0,365,5)[:4])
#    ax.set_xticklabels(["Jan","Apr","Jul","Oct"])
#    ax.set_title(site)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    cb = fig.colorbar(plot,ax=ax,cax=cax)
#    cb.set_ticks(cb.get_ticks()+0.5)
#    cb.set_ticklabels((cb.get_ticks()-0.5).astype(int))
##    plt.clim(-0.5, 5.5)
#    plt.show()    

##############################################################
###find peak month for seasonal locations
#sites_seasonality = pd.read_excel("data/fuel_moisture/seasonality_of_locations.xlsx", index_col = 0, usecols = 2)
#peak_doy = meas.groupby('site').apply(lambda df: df.loc[df['percent'].idxmax(),'meas_date'].dayofyear)
#sites_seasonality.peak_doy = peak_doy
#sites_seasonality.to_excel("data/fuel_moisture/seasonality_of_locations.xlsx") 

##############################################################
### plot FM and SAR with polygon fit 
#good_correlations=[]
#good_correlation_sites=[]    
#for site in meas.site.unique():
#
#    sub = meas.loc[meas.site==site,["meas_date","percent", "fuel"]]
#    obs_sub = obs.loc[obs.Site==site,['obs_date', 'vh_angle_corr']]
#    obs_sub.dropna(inplace = True)
#    if len(sub.fuel.unique())>1 or len(sub)<=25 or len(obs_sub)<=25 or \
#    len(sub.meas_date.dt.year.unique()) < 3 or site not in obs.Site.values:
#        continue
#    counter+=1
#    sub.dropna(inplace =True)
#    sub.sort_values(by ='meas_date', inplace = True)
#    fig, ax = plt.subplots(figsize = (3.7,3))
#    start_year, end_year  = sub.meas_date.dt.year.min(), sub.meas_date.dt.year.max()
#    years = np.arange(start_year, end_year+1)
#    bounds = np.append((np.sort(sub.meas_date.dt.year.unique())),end_year+1)
#
#    plot = ax.scatter(sub.meas_date.dt.dayofyear.values,sub.percent.values, \
#                      color = 'grey', edgecolor = 'w', linewidth=1)
#    ### FM fit
#    meas_fit= np.poly1d(np.polyfit(sub.meas_date.dt.dayofyear.values, sub.percent.values, degree))
#    xd = range(sub.meas_date.dt.dayofyear.min(), sub.meas_date.dt.dayofyear.max())
#    
#    ax.plot(xd, meas_fit(xd), ls = '--', color = 'k', label = 'FM-fit')
#    ### SAR fit
#    
#    obs_fit= np.poly1d(np.polyfit(obs_sub.obs_date.dt.dayofyear.values, \
#                                  obs_sub.vh_angle_corr.values, degree))
#    
#    ax2 = ax.twinx()
#    ax2.scatter(obs_sub.obs_date.dt.dayofyear.values,obs_sub.vh_angle_corr.values, \
#                      color = 'orange', edgecolor = 'w', linewidth=1, marker = '^')
##    xd = range(obs_sub.obs_date.dt.dayofyear.min(), obs_sub.obs_date.dt.dayofyear.max())
##    ax2.plot(xd, obs_fit(xd), ls = '--',color = 'darkorange',label = 'VH-fit')
##    ax2.set_ylim(np.min(obs_fit(xd))*1.05,0.95*np.max(obs_fit(xd)))
#    ma = pd.Series(obs_sub.vh_angle_corr.values, index = obs_sub.obs_date.dt.dayofyear.values)
#    ma = ma.groupby(ma.index).mean()
#    ma = ma.reindex(range(ma.index.min(), ma.index.max()))
#    ma.interpolate(method = 'linear', inplace = True)
#    ma = ma.rolling(60).mean()
#    ma.plot(ls = '--',color = 'darkorange',label = 'VH-fit', ax = ax2)
#    corr = pd.concat([ma, pd.Series(meas_fit(xd), index = xd)],axis = 1)
#    corr = corr.corr().loc[0,1]    
#    if corr>0.5:
#        good_correlations.append(corr)
#        good_correlation_sites.append(site)
#    ax2.set_ylabel("$\sigma_{VH} (dB)$", color = 'orange')
#    ax2.tick_params('y', colors='orange')
#    ax.set_ylabel("Fuel moisture (%)")
##    ax.set_ylabel(r"$\frac{FM - \mu_{site}(FM)}{ \sigma_{site}(FM)}$")
#    ax.set_xlabel("Day of Year")
#    ax.set_xticks(np.linspace(0,365,5)[:4])
#    ax.set_xticklabels(["Jan","Apr","Jul","Oct"])
#    #ax.set_ylim([-2,4])
##    plt.legend()
#
#    
#    plt.show()  
#print(len(good_correlation_sites))
#############################################################
#####  plot fm only with fit
product = "VH"
sites_seasonality = pd.read_excel("fuel_moisture/seasonality_of_locations.xlsx", index_col = 0)
for site in obs.Site.unique():
    sub = obs.loc[obs.Site==site,["obs_date",product]]
    if len(sub)<=25 or site not in sites_seasonality.loc[sites_seasonality.seasonality==1].index:
        continue
    counter+=1
    sub.dropna(inplace =True)
    sub.sort_values(by ='obs_date', inplace = True)

    fig, ax = plt.subplots(figsize = (3,3))
    start_year, end_year  = sub.obs_date.dt.year.min(), sub.obs_date.dt.year.max()
    years = np.arange(start_year, end_year+1)
    bounds = np.append((np.sort(sub.obs_date.dt.year.unique())),end_year+1)
    
    cmap = plt.cm.viridis
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    plot = ax.scatter(sub.obs_date.dt.dayofyear.values,sub[product].values, \
                      c = sub.obs_date.dt.year.values,\
                      cmap =  cmap, label = 'Year', norm = norm, edgecolor = 'grey', linewidth=1)
    ## FM fit
    meas_fit= np.poly1d(np.polyfit(sub.obs_date.dt.dayofyear.values, sub[product].values, degree))
    xd = range(sub.obs_date.dt.dayofyear.min(), sub.obs_date.dt.dayofyear.max())
    ax.plot(xd, meas_fit(xd), ls = '--', color = 'purple', label = 'VH-fit')
    ax.set_ylabel("$\sigma_{VH}$ (dB)")
#    ax.set_ylabel(r"$\frac{FM - \mu_{site}(FM)}{ \sigma_{site}(FM)}$")
    ax.set_xlabel("Day of Year")
    ax.set_xlim(0,365)
    ax.set_xticks(np.linspace(0,365,5)[:4])
    ax.set_xticklabels(["Jan","Apr","Jul","Oct"])
    ax.set_title(site)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(plot,ax=ax,cax=cax)
    cb.set_ticks(cb.get_ticks()+0.5)
    cb.set_ticklabels((cb.get_ticks()-0.5).astype(int))
#    plt.clim(-0.5, 5.5)
    plt.show()   

################################################################################
#### bivariate distribution of mean and range of FM
#def deseasonalize(df):
#    mean_doy= np.poly1d(np.polyfit(df.meas_date.dt.dayofyear.values, df.percent.values,3))
#    df.percent-=mean_doy(df.meas_date.dt.dayofyear.values)
#    return df.percent
#sites_seasonality = pd.read_excel("fuel_moisture/seasonality_of_locations.xlsx", index_col = 0)
#subset_sites = sites_seasonality.loc[sites_seasonality.seasonality==1].index
#meas['meas_year'] = meas.meas_date.dt.year
#sub = meas.loc[meas.site.isin(subset_sites)]
#sub.reset_index(inplace = True)
#sub.loc[:,'percent'] = sub.groupby('site').apply(deseasonalize).reset_index(level=0, drop=True)
##mean_doy= np.poly1d(np.polyfit(sub.meas_date.dt.dayofyear.values, sub.percent.values,3))
##sub.percent-=mean_doy(sub.meas_date.dt.dayofyear)
##plt.plot(range(365), mean_doy(range(365)))
#
#mean = sub.groupby('site').percent.mean()
#sd = sub.groupby('site').percent.std()
#df = pd.concat([mean, sd], axis=1)
#df.columns = ['mean','sd']
#df.sort_values('mean',inplace = True)
#fig, ax = plt.subplots(figsize = (6,3))
#ax.errorbar(range(len(df)), 'mean',yerr = 'sd',data = df, \
#            ecolor = 'grey',marker='s', ms = 3)
#ax.set_ylabel('FM deseasonalized (%)')
#ax.set_xlabel('"Early peak" seasonal sites')
#plt.show()
#print('mean of grey = %0.2f'%df['sd'].mean())
#print('sd of blue = %0.2f '%df['mean'].std())

        
