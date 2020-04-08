# SAR-Enhanced Mapping of Live Fuel Moisture Content

This repository contains analysis performed for the paper ``SAR-Enhanced Mapping of Live Fuel Moisture Content" in the Journal _Remote Sensing of Environment_ by Rao et al., 2020. 
You can view the live fuel moisture content (LFMC) maps produced by the deep learning algorithm in this study in a web-app [here](https://kkraoj.users.earthengine.app/view/live-fuel-moisture).

## Earth Engine Web-app user guide

The [web-app](https://kkraoj.users.earthengine.app/view/live-fuel-moisture) allows users to interactively explore the LFMC maps produced in the paper. The slider bar at bottom controls the time. The blue point on the map controls the location for which LFMC time series is produced from 2016 - 2019. **The web-app has been tested on Google Chrome and Mozilla Firefox only.** It is a known issue that it does not work on Microsoft Internet Explorer or Microsoft Edge. 

Interested in creating your own web-app similar to the Live Fuel Moisture Viewer? [Here](https://code.earthengine.google.com/bb0e411ff41f34149bf459f3960a05e9) is the source code for the web-app. 

### FAQs
1. Can I access more recent LFMC maps? 
As of April 2020, maps from Jan 2016 to Dec, 2019 are available. Moving forward, the project team plans to update the maps directly in the web-app. A fixed update frequency (or a fixed latency) cannot be guaranteed at the moment. For requests related to updating maps, please contact the corresponding author of the manuscript. Do not raise a Github ``issue" for this purpose.
1. Why are there dark green or dark brown patches on some days?
The patches are caused by incorrect cloud or snow masking. The algorithm relies on the in-built quality assessment flags in the Landsat-8 product to mask ``snow'', ``cloud'', or ``cloud shadow''. For more information on how these quality assessment flags were developed refer to [Vermote et al., 2016](https://www.sciencedirect.com/science/article/pii/S0034425716301572).

## Download LFMC maps
![Examples maps of live fuel moisture content created by the deep learning algorithm](/figures/map.png)

The LFMC maps are hosted on Google Earth Engine (GEE) which is a free platform for largescale image visualization and analysis. The maps can be found in an `ee.ImageCollection()` object as a public asset at the following link: https://code.earthengine.google.com/?asset=users/kkraoj/lfm-mapper/lfmc_col. You can use the maps in the following ways-

1. Directly on GEE by importing the collection or 
2. Downloading the maps to your local computer

Both options need a GEE account [signup here](https://earthengine.google.com/). It is free. 

### Use/analyse maps on GEE

1. Once you have your GEE account, open this [script](https://code.earthengine.google.com/6baadb6dc17198d7420eb9df5a4ea4b5)
2. The script will import the LFMC maps as an `ImageCollection` and display the mean for 2019. You can then proceed with your analysis with the imported image collection.

### Download maps to your computer

**Option 1: Code Editor-**

1. Once you have your GEE account, open this [script](https://code.earthengine.google.com/8d145a1cfc6e368fee9d11434867e2cc?noload=true)
1. Modify the `start_date` and `end_date` to suit your needs
1. Modify `scale` to set pixel resolution of output maps. The native resolution of the maps are 250m but you can rescale to whatever resolution you want to suit your analysis. 
1. Click on Run button at the top
1. Click on the Tasks panel on the top right. Verify the maps that you need are set in staging. If ok, click on the Run button beside each map. The maps will be downloaded to your Google Drive in a folder called "lfmc_folder". 

**Option 2: Python API-**

If you want to download many maps, consider using (GEE's python API)[https://developers.google.com/earth-engine/python_install]. It will let you download the maps without having to click the Run button for each map. In the link referred, follow the download instructions.  
1. Once you have the python API installed, open this [script](https://github.com/kkraoj/lfmc_from_sar/blob/master/scripts/download_lfmc.py). 
1. Modify the `start_date` and `end_date` to suit your needs
1. Modify `scale` to set pixel resolution of output maps. The native resolution of the maps are 250m but you can rescale to whatever resolution you want to suit your analysis.
1. Run the script. The maps will be downloaded to your Google Drive in a folder called "lfmc_folder". 

## Repository details

The rest of this Readme file pertains to reproducing the analysis and sharing the algorithms associated with the paper. 

### Scripts:
The repository consists of scripts in the "scripts" folder to perform the following-

1. Prepare input data and train an long-short term memory model to predict LFMC in `LSTM.py`
1. Make plots from the manuscript using `plot_functions.py`
1. Make LFMC maps using `make_map_features_and_predict.py`

Rest of the scripts are not needed. They were used for development of the model and preliminary investigation only.

### Input data

The input data used for predicting LFMC can be found in a pickle object (python 3.6) in the input_data folder. It is a large dataframe with rows corresponding to training examples, and columns corresponding to input features. 

### Trained model

The training model saved using best model checkpoint on keras can be found in trained_model folder. 

## Prerequisites

1. `Python 3.6`
1. `keras v. 2.2.2 `

## Reproducibility guide

1. Clone the repository using `git clone https://github.com/kkraoj/lfmc_from_sar.git`
1. Change the directory addresses of `dir_data` and `dir_codes` in `dirs.py`
1. Run `plot_functions.py` by uncommenting any of the functions at the end of the script to reproduce the figures you wish

## License
Data and scripts presented here are free to use. Please cite the following paper if you use any data or analyses from this study:

[TO BE FILLED AFTER PUBLISHING]

## Issues?

Check the `Issues` tab for troubleshooting or create a new issue.
