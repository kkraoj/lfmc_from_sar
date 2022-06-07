# Set up environment
Type the following in your terminal/anaconda prompt/git bash/command prompt/git

`conda create --name lfmcmaps`
`conda activate lfmcmaps`
`conda install --yes --file requirements.txt`

# Make LFMC maps

1. *Export input feature maps*. This step downloads input landsat 8 and sar maps from gee. Open export_gee_maps.py. In line 85 - 88, choose the appropriate time period for which you want to download the maps. for e.g., `year = 2021`, `day = 1`, and `month in range(1,13)` will download maps for the 1st to 14th of each month in 2021. Run export_gee_maps.py to download sar and opt maps from gee. You can run this from terminal using `python export_gee_maps.py` or from spyder. Running this file will require you to aunthenticate your gee account. Remember which account you used because it will be necessary for the next step.
2. *Download input feature maps* from Google drive. You can either manually go to your Google drive and right click and download the maps, or if there are many maps, use download_and_delete.py. The python file uses Google drive python api which was pretty gnarly to install (had many dependecies, some of which were circular, and the oauth api has changed thrice in this project's lifetime). 
	1. FAQ: While running the python file if this gives a "\n" error, delete token (cookie) from your local machine and try again.
3. *Predict LFMC maps* Run make_map_features_and_predict.py to make LFMC maps using `python vwc_from_radar/codes/scripts/make_map_features_and_predict.py --year 2021`. 
	1. FAQ: Running this file needs scikit-learn 0.19.1. Otherwise the encoder for normalizing data wont load. 
	2. FAQ: Running the file requires 40+ Gb of RAM. The file has been tested on a machine with 64 Gb of RAM. If you encounter out of memory error, try the following.
		1. Close spyder and matlab if open (they get scratch memory allocation when opened). 
		1. Close chrome
		1. Decrease `cache_cutoff` from 1e7 to something smaller. This parameter controls the number of gridcells the model predicts in parallel. At 250 m resolution there are approx. 7e7 gridcells. 		

# Upload LFMC maps
All below steps are for uploading the LFMC maps to earth engine asset. This requires succesful installation of package ``geeup". See requirements.txt for cautionary notes pertaining to this package.
4. Run convert_time_stamp.py. This creates a metadata csv file for earth engine to know which tif file corresponds to which dates.
5. Remove all .ovr and .ovr.xml files from lfmc folder. it should have only .tif
5. open Min GW terminal
6. Open a brand new incognito or a browser window while you are copying cookies, if you have multiple GEE accounts open on the same browser the cookies being copied may create some read issues at GEE end.
7. copy cookies from code.earthengine.google in a fresh browser instance if upload fails with a Unable to read error by clicking on "Copy Cookies" Chrome extension
8. geeup cookie_setup
9. paste the cookie