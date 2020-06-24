# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 01:38:58 2019

@author: kkrao
"""

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from dirs import dir_data
import seaborn as sns


def clean_fmc(df, quality = 'pure species'):
    df = df.copy()
    df.date = pd.to_datetime(df.date)
    df.drop(df[df.percent>=1000].index, inplace = True)
    df = df.loc[df.date.dt.year>=2015]
    df = df.loc[~df.fuel.isin(['1-Hour','10-Hour','100-Hour', '1000-Hour',\
                           'Duff (DC)', '1-hour','10-hour','100-hour',\
                           '1000-hour', 'Moss, Dead (DMC)' ])]
    good_sites = pd.read_excel(os.path.join(dir_data,'fuel_moisture/NFMD_sites_QC.xls'), index_col = 0)
    if quality =='pure species':
        choose = good_sites.loc[(good_sites.include==1)&(good_sites.comment.isin(['only 1'])),'site']
    elif quality =='medium':
        choose = good_sites.loc[(good_sites.include==1),'site']
    elif quality =='all same':
        choose = good_sites.loc[(good_sites.include==1)&(good_sites.comment.isin(['all same'])),'site']
    elif quality =='pure+all same':
        choose = good_sites.loc[(good_sites.include==1)&(good_sites.comment.isin(['only 1', 'all same'])),'site']
    elif quality =='only mixed':
        choose = good_sites.loc[(good_sites.include==1)&(good_sites.comment.isin([np.nan])),'site']
    else :
        raise ValueError('Unknown quality paraneter. Please check input')
    df = df.loc[df.site.isin(choose)]   
    
    return df
    
if __name__ == "__main__":

    sns.set(font_scale = 2, style = 'ticks')
    os.chdir(dir_data)
    smooth_fmc = pd.read_pickle('cleaned_anomalies_11-29-2018/fm_smoothed')
    
    df = pd.read_pickle('vwc')
    ##time filter
    df = df.loc[df.date.dt.year>=2015]
    ## fuel filter
    df = df.loc[~df.fuel.isin(['1-Hour','10-Hour','100-Hour', '1000-Hour',\
                               'Duff (DC)', '1-hour','10-hour','100-hour',\
                               '1000-hour', 'Moss, Dead (DMC)' ])]
    df.drop(df[df.percent>=1000].index, inplace = True)
    
    ## site filter
    fmc = pd.read_csv('fuel_moisture/nfmd_queried_latlon.csv', index_col = 0) 
    zero_lat_lon_sites = fmc[fmc.Latitude==0].index
    df.drop(df[df.site.isin(zero_lat_lon_sites)].index, inplace = True)
    df.index = df.date
    #fmc.loc[(fmc.duplicated(keep = False)==True)&(fmc.Latitude!=0)].to_excel('NFMD_duplicated_sites.xls')
    ### duplicates already dropped in NFMD_sites.xls
    ### ignroe sites for which location data not available
    # df = df.loc[df.site.isin(fmc.Latitude!=0)]
    
    unique_sites = pd.DataFrame(df.site.unique(), columns = ["site"], dtype = 'str')
    unique_sites.sort_values(inplace = True, by = 'site')
    #unique_sites.to_excel("NFMD_sites.xls")
    
    for site in unique_sites['site'][500:600]:
        df_sub = df.loc[df.site==site]
        fig, ax = plt.subplots(figsize = (9, 6))
        for fuel in df_sub.fuel.unique():
            df_sub.loc[df_sub.fuel==fuel].plot(y = 'percent', ax=ax, marker = 'o', label = fuel)
        ax.set_title(site)
        ax.set_ylabel('FMC (%)')
        ax.set_xlabel('')
        ax.legend(bbox_to_anchor = [1,1])
        plt.show()
    

