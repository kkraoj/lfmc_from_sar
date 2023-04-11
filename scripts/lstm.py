# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 00:32:04 2019

@author: kkrao
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle 
from math import sqrt
from numpy import concatenate
from scipy import optimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.offsets import MonthEnd, SemiMonthEnd


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, optimizers
import matplotlib.pyplot as plt

from dirs import dir_data, dir_codes
os.chdir(os.path.join(dir_codes,'scripts'))
from QC_of_sites import clean_fmc
from fnn_smoothed_anomaly_all_sites import plot_pred_actual, plot_importance, plot_usa
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

pd.set_option('display.max_columns', 10)

lc_dict = {14: 'crop',
            20: 'crop',
            30: 'crop',
            50: 'closed broadleaf deciduous',
            70: 'closed needleleaf evergreen',
            90: 'mixed forest',
            100:'mixed forest',
            110:'shrub/grassland',
            120:'grassland/shrubland',
            130:'closed to open shrub',
            140:'grass',
            150:'sparse vegetation',
            160:'regularly flooded forest'}


os.chdir(dir_data)
# convert series to supervised learning


microwave_inputs = ['vv','vh']
optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv']
#optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv','vari','ndii']
mixed_inputs =  ['vv_%s'%den for den in optical_inputs] + ['vh_%s'%den for den in optical_inputs] + ['vh_vv']
dynamic_inputs = microwave_inputs + optical_inputs + mixed_inputs
static_inputs = ['slope', 'elevation', 'canopy_height','forest_cover',\
                    'silt', 'sand', 'clay']

all_inputs = static_inputs+dynamic_inputs
inputs = all_inputs



#%%  
###############################################################################



##input options 
SEED = 0
np.random.seed(SEED)
RELOADINPUT = True
OVERWRITEINPUT = False
LOAD_MODEL = False
OVERWRITE = False
RETRAIN = False
SAVEFIG = False
DROPCROPS = True
RESOLUTION = 'SM'
MAXGAP = '3M'
INPUTNAME = 'lstm_input_data_pure+all_same_28_may_2019_res_%s_gap_%s'%(RESOLUTION, MAXGAP)
# SAVENAME = 'quality_pure+all_same_28_may_2019_res_%s_gap_%s_site_split'%(RESOLUTION, MAXGAP)
SAVENAME = 'quality_pure+all_same_10_apr_2023_res_%s_gap_%s_site_split'%(RESOLUTION, MAXGAP)

##modeling options
EPOCHS = int(20e3)
BATCHSIZE = int(2e5)
DROPOUT = 0.05
TRAINRATIO = 0.80
LOSS = 'mse'
LAG = '3M'
RETRAINEPOCHS = int(5e3)
FOLDS = 3
CV = False

kf = KFold(n_splits=FOLDS, random_state = SEED)
int_lag = int(LAG[0])
if RESOLUTION =='SM':
    int_lag*=2

###############################################################################

#%%modeling

if RELOADINPUT:
    dataset= pd.read_pickle(INPUTNAME)
else:
    if os.path.isfile(INPUTNAME) and not(OVERWRITEINPUT):
        raise  Exception('[INFO] Input File already exists. Try different INPUTNAME')
    dataset, int_lag = make_df(resolution = RESOLUTION, max_gap = MAXGAP, lag = LAG, inputs = inputs)    
    dataset.to_pickle(INPUTNAME)

# ## dropping sar/ratio columns
# for num in ['vh','vv']:
#     for den in ['ndvi','ndwi','nirv']:
#         for col in dataset.columns:    
#             if '%s_%s'%(num, den) in col:
#                 dataset.drop(col, axis = 1, inplace = True)
    
def split_train_test(dataset, inputs = None, int_lag = None, CV = False, fold = None, FOLDS = FOLDS):

    if DROPCROPS:
        crop_classes = [item[0] for item in lc_dict.items() if item[1] == 'crop']
        dataset = dataset.loc[~dataset['forest_cover(t)'].isin(crop_classes)]
    # integer encode forest cover
    encoder = LabelEncoder()
    dataset = dataset.reindex(sorted(dataset.columns), axis=1)
    cols = list(dataset.columns.values)
    for col in ['percent(t)','site','date']:
        cols.remove(col)
    cols = ['percent(t)','site','date']+cols
    dataset = dataset[cols]
    dataset['forest_cover(t)'] = encoder.fit_transform(dataset['forest_cover(t)'].values)
    for col in dataset.columns:
        if 'forest_cover' in col:
            dataset[col] = dataset['forest_cover(t)']
    # normalize features
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset.replace([np.inf, -np.inf], [1e5, 1e-5], inplace = True)
    scaled = scaler.fit_transform(dataset.drop(['site','date'],axis = 1).values)
    rescaled = dataset.copy()
    rescaled.loc[:,dataset.drop(['site','date'],axis = 1).columns] = scaled
    # reframed = series_to_supervised(rescaled, LAG,  dropnan = True)
    reframed = rescaled.copy()
    #### dropping sites with at least 7 training points
#    sites_to_keep = pd.value_counts(reframed.loc[reframed.date.dt.year<2018, 'site'])
#    sites_to_keep = sites_to_keep[sites_to_keep>=24].index
#    reframed = reframed.loc[reframed.site.isin(sites_to_keep)]
    
    print('[INFO] Dataset has %d sites'%len(reframed.site.unique()))
    ####    
    reframed.reset_index(drop = True, inplace = True)
    #print(reframed.head())
     
    # split into train and test sets
    # train = reframed.loc[reframed.date.dt.year<2018].drop(['site','date'], axis = 1)
    # test = reframed.loc[reframed.date.dt.year>=2018].drop(['site','date'], axis = 1)
    #### split train test as 70% of time series of each site rather than blanket 2018 cutoff
    train_ind=[]
    # for site in reframed.site.unique():
    #     sub = reframed.loc[reframed.site==site]
    #     sub = sub.sort_values(by = 'date')
    #     train_ind = train_ind+list(sub.index[:int(np.ceil(sub.shape[0]*TRAINRATIO))])
    if CV:
        for cover in np.sort(reframed['forest_cover(t)'].unique()):
            sub = reframed.loc[reframed['forest_cover(t)']==cover]
            sites = np.sort(sub.site.unique())
            
            if len(sites)<FOLDS:
                train_sites = sites
            else:
                train_sites_ind, _ = list(kf.split(sites))[fold]
                train_sites = sites[train_sites_ind]
                # break
            train_ind+=list(sub.loc[sub.site.isin(train_sites)].index)
        # print(len(train_ind)/reframed.shape[0])
    else:
        for cover in reframed['forest_cover(t)'].unique():
            sub = reframed.loc[reframed['forest_cover(t)']==cover]
            sites = sub.site.unique()
            train_sites = np.random.choice(sites, size = int(np.ceil(TRAINRATIO*len(sites))), replace = False)
            train_ind+=list(sub.loc[sub.site.isin(train_sites)].index)

        
    # sites = reframed.site.unique()
    # train_sites = np.random.choice(sites, size = int(np.ceil(TRAINRATIO*len(sites))), replace = False)
    # train_ind = reframed.loc[reframed.site.isin(train_sites)].index
    
    train = reframed.loc[train_ind].drop(['site','date'], axis = 1)
    test = reframed.loc[~reframed.index.isin(train_ind)].drop(['site','date'], axis = 1)
    train.sort_index(inplace = True)
    test.sort_index(inplace = True)
    #print(train.shape)
    #print(test.shape)
    # split into input and outputs
    train_X, train_y = train.drop(['percent(t)'], axis = 1).values, train['percent(t)'].values
    test_X, test_y = test.drop(['percent(t)'], axis = 1).values, test['percent(t)'].values
    # reshape input to be 3D [samples, timesteps, features]
    if inputs==None: #checksum
        inputs = all_inputs
    train_Xr = train_X.reshape((train_X.shape[0], int_lag+1, len(inputs)), order = 'A')
    test_Xr = test_X.reshape((test_X.shape[0], int_lag+1, len(inputs)), order = 'A')
    return dataset, rescaled, reframed, \
            train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
            scaler, encoder         

dataset, rescaled, reframed, \
    train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
    scaler, encoder = split_train_test(dataset, int_lag = int_lag)
    
# output = open(r'D:\Krishna\projects\vwc_from_radar\data\encoder.pkl', 'wb')
# pickle.dump(encoder, output)
# output.close()

# output = open(r'D:\Krishna\projects\vwc_from_radar\data\scaler.pkl', 'wb')
# pickle.dump(scaler, output)
# output.close()

 
#%% design network

filepath = os.path.join(dir_codes, 'model_checkpoint/LSTM/%s.hdf5'%SAVENAME)

Areg = regularizers.l2(1e-5)
Breg = regularizers.l2(1e-3)
Kreg = regularizers.l2(1e-10)
Rreg = regularizers.l2(1e-15)

def build_model(input_shape=(train_Xr.shape[1], train_Xr.shape[2])):
    
    model = Sequential()
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                  return_sequences=True, \
                  bias_regularizer= Breg))
                  # activity_regularizer = Areg, \
                  
                  # kernel_regularizer = Kreg, \
                  # recurrent_regularizer = Rreg))
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                    return_sequences=True, \
                    bias_regularizer= Breg))
                  # activity_regularizer = Areg, \
                  
                  # kernel_regularizer = Kreg, \
                  # recurrent_regularizer = Rreg))
    # model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
    #                 return_sequences=True, \
    #               activity_regularizer = Areg, \
    #               bias_regularizer= Breg,\
    #               kernel_regularizer = Kreg, \
    #               recurrent_regularizer = Rreg))
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                   bias_regularizer= Breg))
                   # activity_regularizer = Areg, \
                 
                  # kernel_regularizer = Kreg, \
                  # recurrent_regularizer = Rreg))
    # model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True, bias_regularizer= Breg))
    # model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT, bias_regularizer= Breg))#, \
#                   activity_regularizer = Areg, \
#                   bias_regularizer= Breg,\
#                   kernel_regularizer = Kreg, \
#                   recurrent_regularizer = Rreg))
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True))
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT, return_sequences=True))
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True))   
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True))   
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True))
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT))
    #model.add(LSTM(50, input_shape=(train_Xr.shape[1], train_Xr.shape[2]), dropout = 0.3))
    #model.add(LSTM(10, input_shape=(train_Xr.shape[1], train_Xr.shape[2]), dropout = 0.3))
    # model.add(Dense(6))
    model.add(Dense(1))
    optim = optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)
#    optim = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    model.compile(loss=LOSS, optimizer='Nadam')
    # fit network
    return model

checkpoint = ModelCheckpoint(filepath, save_best_only=True)
earlystopping=EarlyStopping(patience=1000, verbose=1, mode='auto')
callbacks_list = [checkpoint, earlystopping]



if  LOAD_MODEL&os.path.isfile(filepath):
    model = load_model(filepath)
 
#    rmse_diff = pd.read_pickle(os.path.join(dir_codes, \
#                'model_checkpoint/LSTM/rmse_diff_%s'%SAVENAME))
    print('[INFO] \t Model loaded')
else:
    if os.path.isfile(filepath):
        if not(OVERWRITE):
            raise  Exception('[INFO] File path already exists. Try Overwrite = True or change file name')
        print('[INFO] \t Retraining Model...')
    model = build_model()

if RETRAIN or not(LOAD_MODEL):
    history = model.fit(train_Xr, train_y, epochs=EPOCHS, batch_size=BATCHSIZE,\
                        validation_data=(test_Xr, test_y), verbose=2, shuffle=False,\
                        callbacks=callbacks_list)
    model = load_model(filepath) # once trained, load best model

    #%% plot history
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='dev')
    ax.set_ylim(0,0.05)
    ax.legend()
    ax.set_xlabel('epochs')
    ax.set_ylabel(LOSS)
    plt.show()
#%% 
#Predictions
def predict(model, test_Xr, test_X, test, reframed, scaler, inputs):
    
    yhat = model.predict(test_Xr)
    #test_X = test_X.reshape((test_X.shape[0], (LAG+1)*test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    pred_frame = test.copy()
    pred_frame.loc[:,:] = scaler.inverse_transform(test.loc[:,:])
    # pred_frame = pred_frame.iloc[:,:len(inputs)+1]
    pred_frame = pred_frame.join(reframed.loc[:,['site','date']])
    inv_y = pred_frame['percent(t)']
    pred_frame['percent(t)_hat'] = inv_yhat
    # calculate RMSE
    rmse = sqrt(mean_squared_error(pred_frame['percent(t)'],pred_frame['percent(t)_hat'] ))
    r2 = r2_score(inv_y, inv_yhat)
    return inv_y, inv_yhat, pred_frame, rmse, r2

inv_y, inv_yhat, pred_frame, rmse, r2  = predict(model, test_Xr, test_X, test, reframed, scaler, inputs)
#%% true vsersus pred scatter
sns.set(font_scale=1.5, style = 'ticks')
plot_pred_actual(inv_y.values, inv_yhat,  np.corrcoef(inv_y.values, inv_yhat)[0,1]**2, rmse, ms = 30,\
            zoom = 1.,dpi = 200,axis_lim = [0,300], xlabel = "Actual LFMC", \
            ylabel = "Predicted LFMC",mec = 'grey', mew = 0, test_r2 = False, bias = True)