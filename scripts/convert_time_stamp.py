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
files = os.listdir(os.path.join(dir_data, 'map/dynamic_maps/lfmc'))
startdates = [file[9:-4] for file in files]
enddates = pd.to_datetime(startdates) + DateOffset(days = -1)
enddates = [x.strftime("%Y-%m-%d") for x in enddates]

for file in files:
    print(file[:-4])
enddates.pop(0)
enddates.append('2019-12-31')

unixtimes = [time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple()) for s in startdates]
for t in unixtimes:
    print(int(t*1000))
    
unixtimes = [time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple()) for s in enddates]
for t in unixtimes:
    print(int(t*1000))