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
import time
import os

import sys
# insert at position 1 in the path, as 0 is the path of this file.
sys.path.insert(1, 'D:/Krishna/projects/google_drive')
from download_and_delete import download, delete_file

## Initialize (a ee python thing)

ee.Initialize()


roi = ee.FeatureCollection('users/kkraoj/west_usa');
#palettes = require('users/gena/packages:palettes');

DISP = True;

band = ['VH','VV'];



#%%/////////////////////////////////////////////////////////////////////////////

def  getQABits(image, start, end, newName):
    #Compute the bits we need to extract.
    pattern = 0;
    for i in range(start, end+1):
       pattern += 2**i;
    
    #Return a single band image of the extracted QA bits, giving the band
    #a new name.
    return image.select([0], [newName]).\
                  bitwiseAnd(pattern).\
                  rightShift(start);

def cloud_shadows(image):
  #Select the QA band.
  QA = image.select(['pixel_qa']);
  #Get the internal_cloud_algorithm_flag bit.
  return getQABits(QA, 3,3, 'Cloud_shadows').eq(0);
  #Return an image masking out cloudy areas.

def snow(image):
  #Select the QA band.
  QA = image.select(['pixel_qa']);
  #Get the internal_cloud_algorithm_flag bit.
  return getQABits(QA, 4,4, 'Snow').eq(0);
  #Return an image masking out snowy areas.

#A function to mask out cloudy pixels.
def clouds(image):
  #Select the QA band.
  QA = image.select(['pixel_qa']);
  #Get the internal_cloud_algorithm_flag bit.
  return getQABits(QA, 5,5, 'Cloud').eq(0);
  #Return an image masking out cloudy areas.
def maskCloudsAndSnow(image):
  cs = cloud_shadows(image);
  c = clouds(image);
  s = snow(image)
  image = image.updateMask(cs);
  image = image.updateMask(s)
  return image.updateMask(c).addBands(image.metadata('system:time_start'));

###Filter by metadata properties.
year = 2022
day=1
# end_date_range = ['%s-%02d-%02d'%(year,month,day) for month in range(1,13)]
end_date_range = ['%s-%02d-%02d'%(year,month,day) for month in range(2,3)]
start_date_range = list((pd.to_datetime(end_date_range) + DateOffset(months = -3)).strftime('%Y-%m-%d'))

downloaded_files = os.listdir("D:/Krishna/projects/google_drive")
out = None
for (start_date, end_date) in zip(start_date_range, end_date_range):
    image_sar = ee.ImageCollection('COPERNICUS/S1_GRD').\
      filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).\
      filterDate(start_date,end_date).\
      filter(ee.Filter.eq('instrumentMode', 'IW')).\
      filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).\
      select(band).mosaic().clip(roi);
    
    
    #Create a cloud-free, most recent value composite.
    recentValueComposite = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').\
      filterDate(start_date,end_date).\
      filterBounds(roi).map(maskCloudsAndSnow).\
      qualityMosaic('system:time_start').clip(roi);
    
    if end_date+'_cloudsnowfree_l8.tif' not in downloaded_files:
        out = batch.Export.image.toDrive(image= recentValueComposite.select(['B2', 'B3', 'B4', 'B5', 'B6']),
          description = end_date+'_cloudsnowfree_l8',
          scale= 250,
          region= roi.geometry().bounds(),
          maxPixels = 1e11)
        batch.Task.start(out)
    if end_date+'_sar.tif' not in downloaded_files:
        out = batch.Export.image.toDrive(
          image=image_sar.select(band),
          description= end_date+'_sar',
          scale= 250,
          region= roi.geometry().bounds(),
          maxPixels=1e11
        );
        batch.Task.start(out)

    # status = out.status()['state']
    
    # while out.status()['state']!="COMPLETED":
    #     time.sleep(60)
    
    # filenames = [end_date+'_cloudsnowfree_l8.tif',end_date+'_sar.tif']
    
    # for filename in filenames:
    #     service, file_id = download(filename)
    #     delete_file(service, file_id)
        
## process the image

out.status()
print("process sent to cloud")