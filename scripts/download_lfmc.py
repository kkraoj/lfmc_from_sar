# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:14:09 2020

@author: kkrao
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 08:30:32 2018

@author: kkrao
"""
import ee
from ee import batch
from pandas.tseries.offsets import DateOffset
import pandas as pd

## Initialize (a ee python thing)

ee.Initialize()

#%%
### Input start and end dates
start_date = '2016-01-01'
end_date =  '2016-01-30'
folder_name = 'lfmc_col' # folder name in GOogle drive where files should be created
scale = 500 #pixel size in meters. lower pixels will consumer more memory and will take longer to download. 

#%%#### create strings for start and end dates

collection = ee.ImageCollection('users/kkraoj/lfm-mapper/lfmc_col').\
                filterDate(start_date,end_date)
                
n = collection.size().getInfo() # number of images to download
    
colList = collection.toList(n)
  
for i in range(n):
    image = ee.Image(colList.get(i));
    id = image.id().getInfo() or 'image_'+i.toString();

    out = batch.Export.image.toDrive(
      image=image,
      folder=folder_name,
      description = id,
      scale= scale,
    );
    batch.Task.start(out)    
## process the image

out.status()
print("process sent to cloud")