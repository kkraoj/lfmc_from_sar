# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 00:32:04 2019

@author: kkrao
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle 

from pandas.tseries.offsets import MonthEnd, SemiMonthEnd

pd.set_option('display.max_columns', 10)

def interpolate(df, var = 'percent', ts_start='2015-01-01', ts_end='2019-05-31', \
                resolution = '1M',window = '1M', max_gap = '4M'):
    df = df.copy()
    df.index = pd.to_datetime(df.date)
    df = df.resample(resolution).mean()

    
    if resolution == '1M':
        interp_limit = int(max_gap[:-1])
    elif resolution =='SM':
        interp_limit = 2*int(max_gap[:-1])
    else:
        raise  Exception('[INFO] RESOLUTION not supported')
    df = df.interpolate(limit = interp_limit)    
    df = df.dropna()
    df = df[var]
    df['date'] = df.index

    return  df

def reindex(df, resolution = '1M'):
    site = df.site.values[0]


    df.index = pd.to_datetime(df.date)
    df = df.resample(rule = resolution, label = 'right' ).mean().dropna()
    df['date'] = df.index
    df['site'] = site
    return df

# convert series to supervised learning

def make_df(resolution = 'SM', max_gap = '3M', lag = '3M', inputs = None):
    ####FMC
    df = pd.read_csv('lfmc.csv', index_col = 0)
    master = pd.DataFrame()
    no_inputs_sites = []
    for site in df.site.unique():
        df_sub = df.loc[df.site==site]
        df_sub = reindex(df_sub,resolution = resolution)
        # no interpolation for LFMC meas. Just reindex to closest 15th day.
        master = master.append(df_sub, ignore_index = True, sort = False)
    ## static inputs    
    static_features_all = pd.read_csv('static_features.csv',dtype = {'site':str}, index_col = 0)
    if not(static_inputs is None):
        static_features_subset = static_features_all.loc[:,static_inputs]
        master = master.join(static_features_subset, on = 'site') 
    ### optical
    
    df = pd.read_csv('opt_500m_cloudless.csv', index_col = 0)
    ### adding VARI and NDII7
#    df['vari'] = (df.green - df.red)/(df.green+df.red-df.blue)
#    df['ndii'] = (df.nir - df['b7'])/(df.nir + df['b7'])
    # for var in optical_inputs:
    opt = pd.DataFrame()
    for site in master.site.unique():
        if site in df.site.values:
            df_sub = df.loc[df.site==site]  
            feature_sub = interpolate(df_sub, var = optical_inputs, resolution = resolution, max_gap = max_gap)
            feature_sub['site'] = site
            opt = opt.append(feature_sub, ignore_index = True, sort = False)
                       
        else:
            if site not in no_inputs_sites:
                print('[INFO]\tsite skipped :\t%s'%site)
                no_inputs_sites.append(site)
        # master = pd.merge(master,feature, on=['date','site'], how = 'outer')         
    ### sar
    df = pd.read_csv('sar_pm_500m.csv', index_col = 0)
    # for var in microwave_inputs:
    micro = pd.DataFrame()
    for site in master.site.unique():
        if site in df.site.values:
            df_sub = df.loc[df.site==site]  
            feature_sub = interpolate(df_sub, var = microwave_inputs, resolution = resolution, max_gap = max_gap)
            feature_sub['site'] = site
            micro = micro.append(feature_sub, ignore_index = True, sort = False)

        else:
            if site not in no_inputs_sites:
                print('[INFO]\tsite skipped :\t%s'%site)
                no_inputs_sites.append(site)
        # master = pd.merge(master,feature, on=['date','site'], how = 'outer')          
    dyn = pd.merge(opt,micro, on=['date','site'], how = 'outer')
    ## micro/opt inputs
    for num in microwave_inputs:
        for den in optical_inputs:
            dyn['%s_%s'%(num, den)] = dyn[num]/dyn[den]
    dyn['vh_vv'] = dyn['vh']-dyn['vv']
    
    dyn = dyn[dynamic_inputs+['date','site']]
    dyn = dyn.dropna()

    ## start filling master
    
    if resolution == '1M':
        int_lag = int(lag[:-1])
    elif resolution =='SM':
        int_lag = 2*int(lag[:-1])
    else:
        raise  Exception('[INFO] RESOLUTION not supported')
        
        
    ##serieal    
    new = master.copy()
    new.columns = master.columns+'(t)'
    
    for i in range(int_lag, -1, -1):
        for col in list(dyn.columns):
            if col not in ['date','site']:
                if i==0:
                    new[col+'(t)'] = np.nan
                else:
                    new[col+'(t-%d)'%i] = np.nan
    for i in range(int_lag, 0, -1):
        for col in list(master.columns):
            if col not in ['date','site','percent']:
                new[col+'(t-%d)'%i] = new[col+'(t)']           
    new = new.rename(columns = {'date(t)':'date','site(t)':'site'})
    count=0        
    for index, row in master.iterrows():
        dyn_sub = dyn.loc[dyn.site==row.site].copy()
        dyn_sub['delta'] = row.date - dyn_sub.date
        if resolution == '1M':
            dyn_sub['steps'] = (dyn_sub['delta'] /np.timedelta64(30, 'D')).astype('int')
        elif resolution =='SM':
            dyn_sub['steps'] = (dyn_sub['delta']/np.timedelta64(15, 'D')).astype('int')
        if all(elem in dyn_sub['steps'].values for elem in range(int_lag, -1, -1)):
            count+=1
            # break debugging
            # print('[INFO] %d'%count)
            dyn_sub = dyn_sub.loc[dyn_sub.steps.isin(range(int_lag, -1, -1))]
            dyn_sub = dyn_sub.sort_values('steps')
            # flat = dyn_sub.stack()
            # flat.index.get_level_values(level=1)
            flat = dyn_sub.pivot_table(columns = 'steps').T.stack().reset_index()
            flat['level_1'] = flat["level_1"].astype(str) + '(t-' +flat["steps"].astype(str)+ ')'
            flat.index = flat['level_1']
            flat.index = flat.index.str.replace('t-0', 't', regex=True)
            flat =  flat.drop(['steps', 'level_1'], axis = 1)[0]
            
            new.loc[index,flat.index] = flat.values
            # print('[INFO] Finding Observation to match measurement... %0.0f %% complete'%(index/new.shape[0]*100))
        sys.stdout.write('\r'+'[INFO] Finding Observation to match measurement... %0.0f %% complete'%(index/new.shape[0]*100))
        sys.stdout.flush()
    new = new.dropna()
    return new , int_lag       


#%%modeling

RESOLUTION = '1M' # SM for semi-monthly or 1M for monthly
MAXGAP = '2M' #If gap between LFMC measurement date and nearest date of sattelile observation is greater than MAXGAP, then ignore that particlar LFMC measurement
LAG = '1M' #number of months of lagged sattelite observations to use corresponding to a single LFMC measurement. For example, if RESULTION = 1M, and LAG = 2M, for every LFMC measurement would be paired with 3 timestamps of sattelite observation (current month + 2 preceding months)

microwave_inputs = ['vv','vh']
optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv']
#optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv','vari','ndii']
mixed_inputs =  ['vv_%s'%den for den in optical_inputs] + ['vh_%s'%den for den in optical_inputs] + ['vh_vv']
dynamic_inputs = microwave_inputs + optical_inputs + mixed_inputs
static_inputs = ['slope', 'elevation', 'canopy_height','forest_cover',\
                    'silt', 'sand', 'clay']

all_inputs = static_inputs+dynamic_inputs
inputs = all_inputs
os.chdir("D:/Krishna/projects/vwc_from_radar/codes/input_data_creation")

dataset, int_lag = make_df(resolution = RESOLUTION, max_gap = MAXGAP, lag = LAG, inputs = inputs)    
dataset.to_csv("lfmc_with_features.csv")

