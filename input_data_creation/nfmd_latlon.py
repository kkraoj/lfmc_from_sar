# -*- coding: utf-8 -*-
"""
Created on Mon May 28 23:26:24 2018

@author: kkrao
"""
import os
import requests
import pdb
from bs4 import BeautifulSoup
import pandas as pd


dir_data = "D:/Krishna/projects/vwc_from_radar/data/fuel_moisture"
os.chdir(dir_data)
query = pd.read_csv('site_info_query_10-16-2018.csv', index_col = 0)
for col in query.columns:
    query[col] = query[col].str.replace(' ','%20')


df = pd.DataFrame()

for (_, [gacc, state, group, site]) in query.iterrows():
    url_to_scrape = ("https://www.wfas.net/nfmd/include/site_page.php?"
                    "site={site}&"
                    "gacc={gacc}&"
                    "state={state}&"
                    "grup={group}").format(site = site, gacc = gacc, 
                                    state = state, group = group)
#    print(site, gacc, group, state)
    r = requests.get(url_to_scrape)
    if r.status_code !=200:
        print('[INFO] Web site not found for %s'%site)
        continue
    else:
        print('[INFO] Web site found for %s'%site)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find('table')
    rows = soup.find_all('tr')
    for tr in rows:
        cols = tr.find_all('td')
        if cols[0].text == 'Location':
#            pdb.set_trace()
#            print(1)
            lat = cols[1].text.split('x')[0][:-1]
            lon = cols[1].text.split('x')[1][1:]
#                
            temp = pd.DataFrame([[lat,lon]], columns = ['latitude','longitude'], index = [site])
            df = pd.concat([df, temp], axis = 0)
            if '-' in lat:
                print('[INFO] location found for %s'%site)
            else:
                print(url_to_scrape)
            break
### manually filled all KNF sites in Arizona
df.index = df.index.str.replace('%20', ' ')

df.loc[df.latitude == "",'latitude'] = '0-0-0'
df.loc[df.longitude == "",'longitude'] = '0-0-0'
df.latitude = [float(e[0])+float(e[1])/60+float(e[2])/3600 for e in df.latitude.str.split('-')]
df.longitude = [float(e[0])+float(e[1])/60+float(e[2])/3600 for e in df.longitude.str.split('-')]
df.longitude *= -1
df.index.name = 'site'
df.to_csv("nfmd_queried_latlon_10-16-2018.csv")
