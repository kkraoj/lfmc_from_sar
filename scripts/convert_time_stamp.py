# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:55:36 2020

@author: kkrao
"""

import numpy as np
import pandas as pd
import os
from dirs import dir_data, dir_codes,dir_figures
import time
import datetime
from pandas.tseries.offsets import DateOffset

########################perparing metadata for geebam upload
df = pd.DataFrame(index =  os.listdir(os.path.join(dir_data, 'map/dynamic_maps/lfmc')))
df.index.name = 'filename'

startdates = [file[9:-4] for file in df.index]
enddates = pd.to_datetime(startdates) + DateOffset(days = -1)
enddates = [x.strftime("%Y-%m-%d") for x in enddates]

# for file in files:
#     print(file[:-4])
enddates.pop(0)
enddates.append('2020-07-14')

df['system:time_start'] = [time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple())*1000 for s in startdates]

    
df['system:time_end']= [time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple())*1000 for s in enddates]
df.tail()
df.to_csv("D:/Krishna/projects/vwc_from_radar/gee-app/upload_meta_data _30_jun_2020.csv")