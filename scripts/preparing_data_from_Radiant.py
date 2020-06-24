# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:47:55 2020

@author: kkrao
"""


import pandas as pd
import os 
from dirs import dir_data, dir_codes


RESOLUTION = '1M'
MAXGAP = '3M'

INPUTNAME = 'lstm_input_data_pure+all_same_28_may_2019_res_%s_gap_%s'%(RESOLUTION, MAXGAP)

latlon = pd.read_csv(dir_data+"/fuel_moisture/nfmd_queried_latlon.csv", index_col = 0)
latlon.index.name = "site"
latlon.columns = latlon.columns.str.lower()

dataset= pd.read_pickle(os.path.join(dir_codes,'input_data',INPUTNAME))
dataset = dataset.join(latlon, on = "site")

dataset.to_csv(os.path.join(dir_codes,'input_data',"training_features.csv"))

describe = pd.DataFrame({'column_name':dataset.columns})
describe['description'] = ""

# describe.to_csv(os.path.join(dir_codes,'input_data',"training_features_description.csv"))

