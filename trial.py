# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 02:46:03 2018

@author: kkrao
"""

import pandas as pd

a = pd.Series(range(2,5), index = range(2,5))
b = pd.Series(range(10))
(a-b).dropna()
