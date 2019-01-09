# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 02:46:03 2018

@author: kkrao
"""
import os

import pandas as pd
import numpy as np
from dirs import dir_data


os.chdir(dir_data)
meas_sub = pd.read_pickle('for_investigation_corrupt_fm')
meas_sub.percent.astype(np.int).plot(marker = 'o', ms = 3, ls='-', mfc = "none",mew = 0.2,  c = "maroon",mec = "maroon", label = 'fm')

