'''

#!/usr/bin/env python

# Google Earth Engine (GEE) Subset package
# 
# Easy subsetting of remote sensing
# time series for processing external
# to GEE.
# 
# This in parts replaces the ORNL DAAC
# MODIS subsets, but extends it to higher
# resolution date such as Landsat and
# Sentinel. It should also work on all
# other gridded products using the same
# product / band syntax.

# load required libraries

to run:

python download_sentinel.py -p "COPERNICUS/S1_GRD" \
                     -b "VH" \
                     -f "nfmd_queried_trial.csv" \
                     -d "500m_ascending_VH_9-9-2020" \
                     -sc 500 \
                     -s 2019-06-01 \
                     -e 2020-02-27

###########################################################################

product = "COPERNICUS/S1_GRD"
bands = ["VH"]
#args.directory = "D:\Krishna\projects\vwc_from_radar\data\sar\new"
start_date = "2018-01-01"
end_date = "2018-01-05"
latitude = 48.09888889
longitude = -120.1425
pad = 0.05
site = "Hungry Hunter 33_42"
scale = 30

'''
import os, argparse
import pandas as pd
import ee
import time
from datetime import datetime, timedelta
from collections import OrderedDict
from calendar import monthrange


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 200)

today = datetime.now()
today = today.strftime("%Y-%m-%d")
# parse arguments in a beautiful way
# includes automatic help generation
def getArgs():

   # setup parser
   parser = argparse.ArgumentParser(
    description = '''Google Earth Engine subsets script: 
                    Allows for the extraction of remote sensing product
                    time series for most of the data available on
                    Google Earth Engine. Locations need to be specied as
                    either a comma delimited file or an explicit latitude
                    and longitude using command line arguments.''',
    epilog = '''post bug reports to the github repository''')
   parser.add_argument('-p',
                       '--product',
                       help = 'remote sensing product available in GEE',
                       required = True)
                       
   parser.add_argument('-b',
                       '--bands',
                       help = 'band name(s) for the requested product',
                       nargs = "+",
                       required = False)
                       
   parser.add_argument('-s',
                       '--start',
                       help = 'start date of the time series (yyyy-mm-dd)',
                       default = "2014-01-01")
                       
   parser.add_argument('-e',
                       '--end',
                       help = 'end date of the time series (yyyy-mm-dd)',
                       default = today)
                       
   parser.add_argument('-pd',
                       '--pad',
                       help = '''grow sampling location 
                       in km east west north south''',
                       default = 0,
                       type = float)

   parser.add_argument('-sc',
                       '--scale',
                       help = '''scale in meter, match the native resolution of
                       the data of interest otherwise mismatches in scale will result in
                       high pixel counts and a system error''',
                       default = "30")

   parser.add_argument('-l',
                       '--location',
                       nargs = 2,
                       help = '''geographic location as latitude longitude
                       provided as -loc latitude longitude''',
                       default = 0,
                       type = float)

   parser.add_argument('-f',
                       '--file',
                       help = '''path to file with geographic locations
                        as provided in a csv file''',
                       default = 0)
                       
   parser.add_argument('-d',
                       '--directory',
                       help = '''directory / path where to write output when not
                       provided this defaults to output to the console''',
                       default = 0)  
                                        
   parser.add_argument('-v',
                       '--verbose',
                       help = '''verbose debugging''',
                       default = False)  
   # put arguments in dictionary with
   # keys being the argument names given above
   return parser.parse_args()
def reduceNeighborhood(image):
    return ee.Image.reduceNeighborhood(image, 
                          reducer=ee.Reducer.mean(),
                          kernel=ee.Kernel.circle(7)
                          )
def reduceRegion(image):
    return ee.Image.reduceRegion(image, 
                          reducer=ee.Reducer.mean()
                          )
# GEE subset subroutine 
def gee_subset(product = None,
              bands = None,
              start_date = None,
              end_date = None,
              latitude = None,
              longitude = None,
              scale = None,
              pad = 0,
              site = None):

   # fix the geometry when there is a radius
   # 0.01 degree = 1.1132 km on equator
   # or 0.008983 degrees per km (approximate)
   if pad > 0 :
    pad = pad * 0.008983 

   # setup the geometry, based upon point locations as specified
   # in the locations file or provided by a latitude or longitude
   # on the command line / when grow is provided pad the location
   # so it becomes a rectangle (report all values raw in a tidy
   # matrix)

   if pad:
     geometry = ee.Geometry.Rectangle(
       [longitude - pad, latitude - pad,
       longitude + pad, latitude + pad])
   else:
     geometry = ee.Geometry.Point([longitude, latitude])

   # define the collection from which to sample

   col = ee.ImageCollection(product).\
    filterDate(start_date, end_date).\
    filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).\
    filter(ee.Filter.listContains('transmitterReceiverPolarisation', bands[0]))
    
    

#    select(tuple(bands))
    
    

#    filter(ee.Filter.eq('instrumentMode', 'IW')).\
#    select(tuple(bands)).\



   # region values as generated by getRegion
   try:
      region = col.getRegion(geometry, int(scale)).getInfo()
   except:
      return pd.DataFrame()
#       print('Value = %0.2f'%region[1][4])
   # stuff the values in a dataframe for convenience      
   df = pd.DataFrame.from_records(region[1:len(region)])
   if df.shape == (0,0):
      return pd.DataFrame()
   else: 
      # use the first list item as column names
      df.columns = region[0]
      
      # drop id column (little value / overhead)
      df.drop('id', axis=1, inplace=True)
      
      # divide the time field by 1000 as in milliseconds
      # while datetime takes seconds to convert unix time
      # to dates
      df.time = df.time / 1000
      df['time'] = pd.to_datetime(df['time'], unit = 's')
      df.rename(columns = {'time': 'date'}, inplace = True)
      df.sort_values(by = 'date')
      # add the product name and latitude, longitude as a column
      # just to make sense of the returned data after the fact
#      df['product'] = pd.Series(product, index = df.index)
         
      # return output
      return df


def split_years(start_date, end_date):
    """
    Function to split query period to bypass computation timed out (300s query limit of gee)
    Takes input as start and end years
    Gives output as list of start and end periods for each year
    """
  
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in [start_date, end_date]]
    start = list(OrderedDict(((start + timedelta(_)).strftime(r"%Y-%m-01"), None) for _ in range((end - start).days)).keys())
    end = [item[:-2]+str(monthrange(int(item[:4]),int(item[5:7]))[1]) for item in start] 
#    print(end)
    return zip(start,end)
    
if __name__ == "__main__":

   # parse arguments
    args = getArgs()
   
   # read in locations if they exist,
   # overrides the location argument
    if args.file:
        if os.path.isfile(args.file):
            locations = pd.read_csv(args.file)
            print("[INFO] Read input locations")
        else:
            print("[INFO] Not a valid location file, check path")
    elif args.location:
        locations = pd.DataFrame(['site'] + args.location).transpose()  	 
   
   # initialize GEE session
   # requires a valid authentication token
   # to be present on the system
    ee.Initialize()
   
   # now loop over all locations and grab the
   # data for all locations as specified in the
   # csv file or the single location as specified
   # by a lat/lon tuple
   
    for loc in locations.itertuples():
        df = pd.DataFrame()    
        start_time = time.time()
       # download data using the gee_subset routine
       # print to console if verbose
        for (start, end) in [(args.start, args.end)]:
            try:
                df_sub = gee_subset(product = args.product,
                               bands = args.bands,
                               start_date = start,
                               end_date = end,
                               latitude = loc[2],
                               longitude = loc[3],
                               scale = args.scale,
                               pad = args.pad,
                               site = loc[1])
                df = pd.concat([df, df_sub], axis = 0)  
    
            except NameError:
                print("Error: check input parameters")
                if args.verbose:
                    raise        
        elapsed_time = (time.time()-start_time)
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if df.shape == (0,0):
            print('[INFO]\t'
                  '[{0}/{1}]\t'
                  'Time Taken {2}\t'
                  'Data not available for site {3}'.format(
                          loc.Index+1, locations.shape[0],elapsed_time, loc[1]))
        else:
            if args.directory:
                if not(os.path.isdir(args.directory)):
                    os.mkdir(args.directory) 
                df.to_csv(os.path.join(args.directory,"%s_gee.csv"%loc[1]), index = False)  
                print('[INFO]\t'
                      '[{0}/{1}]\t'
                      'Time Taken {2}\t'
                      'Data processed for site {3}'.format(
                              loc.Index+1, locations.shape[0],elapsed_time,loc[1])) 
            else:
                print(df)
 