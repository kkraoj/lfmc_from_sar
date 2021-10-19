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
import glob
import calendar

########################perparing metadata for geebam upload
filenames =  glob.glob(os.path.join(dir_data, 'map/dynamic_maps/lfmc', '*.tif'))

filenames = [filename[-4-10-4-5:-4] for filename in filenames]
df = pd.DataFrame(index = filenames)
df.index.name = 'id_no' # must be id_no. Dont change

startdates = [file[9:] for file in df.index]
startDatesTemp = startdates.copy()


year = startdates[-1][:4]
month = startdates[-1][5:7]
day = startdates[-1][-2:]

if day=="01":
    toAppend = year+"-%s-15"%(month)
    startDatesTemp.append(toAppend)
else:
    toAppend = (pd.to_datetime(startdates[-1]).replace(day=1)+ datetime.timedelta(days=32)).replace(day=1).strftime("%Y-%m-%d")
    startDatesTemp.append(toAppend)

enddates = pd.to_datetime(startDatesTemp) + DateOffset(days = -1)
enddates = [x.strftime("%Y-%m-%d") for x in enddates]

enddates.pop(0)
# enddates.append('2020-10-14')



df['system:time_start'] = [calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple())*1000 for s in startdates]
df['system:time_end']= [calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple())*1000 for s in enddates]
df.tail()
df.to_csv("D:/Krishna/projects/vwc_from_radar/gee-app/upload_meta_data.csv")
