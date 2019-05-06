# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:58:38 2019

@author: kkrao
"""

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from dirs import dir_data

os.chdir(dir_data)
pd.set_option('display.max_columns', 30)
##load df with optimized weights for each species for all sites calculated in 
## fnn_predict.py
df = pd.read_pickle('mixed_species/optimization_results')   
df.head()
