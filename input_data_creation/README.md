# Guide to create input features (sattelite and map-based) to model LFMC

The overall process flow consists of obtaining some LFMC data along with latitude and longitude, downloading sattelite data corresponding to these locations and times, and extracting static data (like landcover) from maps, and finally attanging them in rows (examples) and columns (features). 
Note: "Input features" from here onwards refers to the features that will help you predict LFMC. For example, landsat-8 bands are one of the input data. 

Requirements: Python 3.6+, `gdal` and [Google Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)

1. List of input features is given in Table 1 in the [main manuscript.](https://www.sciencedirect.com/science/article/pii/S003442572030167X)
1. Create a .csv file with latitude and longitude of locations for which you want to create input features. These are ideally a list of locations for which you already have ground measurement of LFMC (and thus, you can train a model based on the data). A sample csv file with latitude and longitude of all LFMC sampling stations in the western USA found in the [National Fuel Moisture Database](http://www.wfas.net/index.php/national-fuel-moisture-database-moisture-drought-103) can be found in `nfmd_queried_latlon.csv`. This sheet was created using the `script nfmd_laton.py` and the lfmc measurements were downloaded using `nfmd_download.py` and rearranged into a pandas dataframe using `nfmd_compile.py`. 
1. Download dynamic features: To download dynamic features like Sentinel-1 SAR backscatter, Landsat-8 optical reflecatance, use `download_sentinel.py` or `download_landsat.py`. These scripts can be modified to download any data available as an Image Collection in Earth Engine by substituting the necessary Image Collection Id. This script (run from the command line) will download time series of sattelite data of your choice at the locations specified in the input csv file (created in the above step). This script requires Google Earth Engine's python API. For example, if you wish to download landsat-8 data navigate to the `create_input_data` folder in the terminal and type something like: 
	```
	python download_landsat.py -p LANDSAT/LC08/C01/T1_SR \
	                -b B1 B2 B3 B4 B5 B6 B7 pixel_qa  \
	                -s "2015-01-01" \
	                -e "2021-10-21" \
	                -f "nfmd_queried_latlon.csv" \
	                -sc 500 \
	                -d "L8_500m"
	```
	Similarly, to download sentinel-1 backscatter, use
	```
	python download_sentinel.py -p "COPERNICUS/S1_GRD" \
                     -b "VH" \
                     -f "nfmd_queried_latlon.csv" \
                     -d "S1_500m" \
                     -sc 500 \
                     -s "2015-01-01" \
	                 -e "2021-10-21"
	```
	Note how the script downloads data for a period of ~8 months (from `-s` to `-e`). These periods can be adjusted based on the date of LFMC measurements. It may help to extend the input features into the past as prior sattelite data may be useful to model current LFMC.
1. Rearrange dynamic features: Use `make_features.py` to rearrange the dyanmic features into 1 csv file.  
1. Static features: Extract static features like topography, soil texture, landcover, canopy height etc. from static maps at the locations specified in the csv file mentioned above using `make_features.py`. Static maps of diffferent variables like soil texture, land cover, etc. for USA were obtained from publicly available GeoTiffs. A list of sources is provided in the Methods sections of the manuscript. If you don't wish to download them from each source on your own, and want to request the authors of the manuscript to share their downloaded versions, please contact the corresponding author of the manuscript. 
1. Combine dynamic features, static features, and the LFMC measurements into a single dataframe to allow model training using `make_dataframe.py`.
