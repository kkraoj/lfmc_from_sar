# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:39:15 2018

@author: kkrao
"""

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dirs import dir_data

os.chdir(dir_data)

##### saving ppt in pickle
#df = pd.DataFrame()
#for file in os.listdir('ppt/raw/'):
#    df = df.append(pd.read_csv(os.path.join('ppt/raw/',file), skiprows = 10), ignore_index = True)
#df.rename(columns = {'Name':'site'}, inplace = True)
#df.columns = df.columns.str.lower()
#df.to_pickle('ppt/prism_ppt')
###############################################################################
##### plot sar and ppt timeseries


### plotting ts
pass_type = 'pm'
obs = pd.read_pickle("sar_%s_500m"%pass_type)
obs.index = obs.date
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
##opt.interpolate(method = "linear",inplace = True)
ppt= pd.read_pickle("ppt/prism_ppt")
ppt.date = pd.to_datetime(ppt.date)
ppt.index = ppt.date
site_lookup = pd.read_csv("ppt/nfmd_queried_latlon_all.csv")
pure_species_sites = ['Clark Motorway, Malibu',
                     'Corralitos New Growth',
                     'Corralitos Old Growth',
                     'Glendora Ridge, Glendora',
                     'Mt. Woodson Station',
                     'Pebble Beach New Growth',
                     'Pebble Beach Old Growth',
                     'Ponderosa Basin Ceanothus, Buckbrush New', 
                     'Ponderosa Basin Ceanothus, Buckbrush Old',
                     'Ponderosa Basin Manzanita, Whiteleaf New',
                     'Ponderosa Basin Manzanita, Whiteleaf Old',
                     'ST RTE 35 & 92',
                     'Tapo Canyon, Simi Valley',
                     'Throckmorton',
                     'Trippet Ranch, Topanga']

c1 = 'y'
c2 = 'deepskyblue'
ctr = 0
var = "vv"
sites=[]
for site in pure_species_sites:
    obs_sub = obs.loc[obs.site==site,:]
    if len(obs_sub.loc[obs_sub.index<='2017-01-01',:])<10:
        continue
    if site not in site_lookup.site.values:
        continue
    lat,lon = np.round(site_lookup.loc[site_lookup.site==site,['latitude','longitude']].values[0],4)
    if lat not in ppt.latitude.values:
        continue
    ppt_sub = ppt.loc[(ppt.latitude==lat)&(ppt.longitude==lon),:]
    fig, ax = plt.subplots(figsize = (4,1.5))
    obs_sub[var].plot(ax = ax, marker = 'o', ms = 3, ls='-', c= c1,mfc = "none",mew = 0.5,  mec = c1, label = 'loess')
#    obs_sub[var].resample('1d').asfreq().interpolate('linear').plot(ax = ax, marker = 'None', ms = 3, ls='-', c= c1,mfc = "none",mew = 0.5,  mec = "y", label = 'loess')
    ax.set_ylabel('$\sigma_{VH}$(dB)', color = c1)
    ax.set_title(site)
    ax.tick_params('y', colors=c1)
    ax.set_xlim(obs_sub.index.min()), obs_sub.index.max()
    
    ax2 = ax.twinx()
    ax2.tick_params('y', colors=c2)
    ax2.set_ylabel('ppt (mm)', color = c2)
    ax2.bar(x = ppt_sub.index,height = ppt_sub['ppt (mm)'].values,   color =c2, label = 'ppt')
#    ax.xaxis.set_major_locator(mdates.YearLocator())
    #set major ticks format
#    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
#    ax3 = ax.twinx()
#    
#    opt_sub.b3.plot(ax=ax3,  marker = 'o', ms = 2, ls='none', mfc = "none",mew = 0.5,  mec = "g", label = 'loess')
#    ax3.plot(x3_,y3_, color = 'g', lw = 1)
#    ax3.set_ylabel('Green', color='g')
#    ax3.tick_params('y', colors='g')
#    ax3.spines["right"].set_position(("axes", 1.15))
#
#    ax.set_xlabel("")
#    ax.annotate(r'FM (%)', color = 'maroon', xy=(1.0, 1.07), xycoords='axes fraction')
    plt.show()
#    ctr+=1
#    sites.append(site)
#print('Total sites = %s'%ctr)