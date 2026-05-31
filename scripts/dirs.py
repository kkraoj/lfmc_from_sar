# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:34:43 2018

@author: kkrao
"""

import os

if os.environ.get('SHERLOCK'):
    dir_data    = "/oak/stanford/groups/konings/projects/rao_2020/data"
    dir_codes   = "/oak/stanford/groups/konings/projects/rao_2020/code/lfmc_from_sar/codes"
    dir_figures = "/oak/stanford/groups/konings/projects/rao_2020/data/figures"
else:
    dir_data    = "D:/Krishna/projects/vwc_from_radar/data"
    dir_codes   = "D:/Krishna/projects/vwc_from_radar/codes"
    dir_figures = "D:/Krishna/projects/vwc_from_radar/figures"