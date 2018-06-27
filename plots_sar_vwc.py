# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:27:49 2018

@author: kkrao
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from scipy import stats

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2, stats.pearsonr(x, y)[1]

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 200)
sns.set(font_scale = 1, style = 'ticks')


os.chdir('D:/Krishna/projects/vwc_from_radar')
df = pd.read_pickle('data/df_sar_vwc')
#df.obs_date_local = pd.to_datetime(df.obs_date_local, format = '%Y-%m-%d %H:%M')
###############################################################################
df = df.loc[df.meas_date>='2016-01-01',:]
#df = df.loc[(df.residual.abs() <=14),:]
print(df.shape)
fig, ax = plt.subplots()

df.hist(column = 'residual', bins = 250, facecolor='g',edgecolor='black',  alpha=0.75, ax = ax)
ax.set_xlabel('$\Delta$ delay (days)')
ax.set_ylabel('Frequency (-)')
ax.set_title('')
ax.grid('off')
ax.set_xlim([-14,14])
#ax.axvline(x=-5, ls = '--', c='r')
#ax.axvline(x=5, ls = '--', c='r')
###############################################################################
#filter = (df.site == '2205')
#
#d = df.loc[filter,:]
#
#fig, ax = plt.subplots()
#d.plot(x = 'meas_date', y = ['2in','4in','8in','20in','40in'], style='.', ax = ax)
#ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
###############################################################################
#df.loc[df.obs_date_local.dt.hour >15,:].shape[0]/df.shape[0]
#ax = df.obs_date_local.apply(lambda x: x.hour).hist()
#ax.set_xlabel('Local Time of Day')
#ax.set_ylabel('Frequency')
###############################################################################
#filter =
#    (df.residual.abs() <=10) \
#    &(df.obs_date_local.dt.hour>17)\
#    &(df.obs_date_local.dt.year>=2016)\
#            &(df.obs_date_local.dt.month>=6)\
#            &(df.obs_date_local.dt.month<9)
#d = df.loc[filter,:]
#
#fig, ax = plt.subplots()
##d.plot.scatter(x = 'p2in', y = 'VV', \
##              ylim = [-20,-5], xlim = [-10**4, -10**(-1)], ax = ax)
#ax.set_xscale('symlog')
##ax.set_yscale('symlog')
#sns.regplot(x="p20in", y="VV", data=d, ax = ax, order = 1, color = 'darkgreen')
#ax.set_ylabel('VV (dB)')
#ax.set_xlabel('$\psi$ (KPa)')
#ax.set_title('Trees')
#ax.set_ylim([-20,-5])
#ax.set_xlim([-10**2.5, -10**0])
#r2 = d['p20in'].corr(d['VV'])**2
#ax.annotate('$R^2$ = %0.2f'%r2, xy=(0.65, 0.15), xycoords='axes fraction')
#
#y=d['VV']; x=d['p8in']
#non_nan_ind=np.where(~np.isnan(x))[0]
#x=x.take(non_nan_ind);y=y.take(non_nan_ind)
#non_nan_ind=np.where(~np.isnan(y))[0]
#x=x.take(non_nan_ind);y=y.take(non_nan_ind)
#stats.pearsonr(x, y)


###############################################################################
#for site in [sites_20in[2]]:
#    filter =\
#            (df.site == site)\
#            &(df.residual.abs() <=6) \
#            &(df.obs_date_local.dt.hour>17)\
#            &(df.obs_date_local.dt.year>=2016)\
##            &(df.obs_date_local.dt.month>=6)\
##            &(df.obs_date_local.dt.month<9)
#
#    d = df.loc[filter,:]
#    if d.shape[0]<30:
#        continue
#    fig, ax = plt.subplots()
#    ax.plot(d.obs_date_local, d['p2in'].rolling(4).mean(), 'b.')
#    ax.set_yscale('symlog')
#    ax.set_ylim([-10**4, -10**(-1)])
#    ax.set_ylabel('$\psi (KPa) $', color='b')
#    ax.tick_params('y', colors='b')
##    fmt = '-{x:, .0f}'
##    tick = mtick.StrMethodFormatter(fmt)
##    ax.yaxis.set_major_formatter(tick) 
##    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,-1))
#    ax2 = ax.twinx()
#    ax2.plot(d.obs_date_local, d.VV.rolling(3).mean(), 'm.')
#    ax2.set_ylim([-30,0])
#    ax2.set_ylabel('VV (dB)', color = 'm')
#    ax2.tick_params('y', colors='m')
#    fig.autofmt_xdate()
##    d.plot(x = 'obs_date_local', y = ['8in','VV'], style=['-','--'], ax = ax)
##    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
##    ax.set_title('Site %s'%site)
##    ax.set_xlim([-25,0])
#    plt.show()
    
###############################################################################