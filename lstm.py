# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 00:32:04 2019

@author: kkrao
"""

import os
import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from dirs import dir_data, dir_codes
os.chdir(dir_codes)
from QC_of_sites import clean_fmc
from fnn_smoothed_anomaly_all_sites import plot_pred_actual, plot_importance, plot_usa
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

pd.set_option('display.max_columns', 10)

def interpolate(df, var = 'percent', ts_start='2015-01-01', ts_end='2018-12-31', window = '30d', max_gap = '120d'):
    df = df.copy()
    df.index = df.date
    df.dropna(subset = [var], inplace = True)
    x_ = df.groupby(df.index).mean().resample('1d').asfreq().index.values
    y_ = df.groupby(df.index).mean().resample('1d').asfreq()[var].interpolate().rolling(window = window).mean()
    z = pd.Series(y_,x_)
    df.sort_index(inplace = True)
    df['delta'] = df['date'].diff()
    gap_end = df.loc[df.delta>=max_gap].index
    gap_start = df.loc[(df.delta>=max_gap).shift(-1).fillna(False)].index
    for start, end in zip(gap_start, gap_end):
        z.loc[start:end] = np.nan
    z =z.reindex(pd.date_range(start=ts_start, end=ts_end, freq='1M'))
    z = pd.DataFrame(z)
    z.dropna(inplace = True)
    z['site'] = df.site[0]
    z['date'] = z.index
    return  z

os.chdir(dir_data)
# convert series to supervised learning


microwave_inputs = ['vv','vh']
optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv']
mixed_inputs =  ['vv_%s'%den for den in optical_inputs] + ['vh_%s'%den for den in optical_inputs] + ['vh_vv']
dynamic_inputs = microwave_inputs + optical_inputs + mixed_inputs

static_inputs = ['slope', 'elevation', 'canopy_height','forest_cover',\
                    'silt', 'sand', 'clay']

all_inputs = static_inputs+dynamic_inputs

def make_df():
    ####FMC
    df = pd.read_pickle('fmc_04-29-2019')
    df = clean_fmc(df, quality = 'pure+all same')
    master = pd.DataFrame()
    no_inputs_sites = []
    for site in df.site.unique():
        df_sub = df.loc[df.site==site]
        df_sub = interpolate(df_sub, 'percent')
        master = master.append(df_sub, ignore_index = True, sort = False)
     
    ### optical
    df = pd.read_pickle('landsat8_500m_cloudless')
    for var in optical_inputs:
        feature = pd.DataFrame()
        for site in master.site.unique():
            if site in df.site.values:
                df_sub = df.loc[df.site==site]  
                feature_sub = interpolate(df_sub, var)
                feature = feature.append(feature_sub, ignore_index = True, sort = False)
            else:
                if site not in no_inputs_sites:
                    print('[INFO]\tsite skipped :\t%s'%site)
                    no_inputs_sites.append(site)
        master = pd.merge(master,feature, on=['date','site'], how = 'outer')         
    ### sar
    df = pd.read_pickle('sar_ascending_30_apr_2019')
    for var in microwave_inputs:
        feature = pd.DataFrame()
        for site in master.site.unique():
            if site in df.site.values:
                df_sub = df.loc[df.site==site]  
                feature_sub = interpolate(df_sub, var)
                feature = feature.append(feature_sub, ignore_index = True, sort = False)
            else:
                if site not in no_inputs_sites:
                    print('[INFO]\tsite skipped :\t%s'%site)
                    no_inputs_sites.append(site)
        master = pd.merge(master,feature, on=['date','site'], how = 'outer')          
    
    ## static inputs    
    static_features_all = pd.read_csv('static_features.csv',dtype = {'site':str}, index_col = 0)
    if not(static_inputs is None):
        static_features_subset = static_features_all.loc[:,static_inputs]
        master = master.join(static_features_subset, on = 'site')
    
    ## micro/opt inputs
    for num in microwave_inputs:
        for den in optical_inputs:
            master['%s_%s'%(num, den)] = master[num]/master[den]
    master['vh_vv'] = master['vh']/master['vv']
    master.reset_index(drop = True, inplace = True)
    return master    
        


# convert series to supervised learning
def series_to_supervised(df, n_in=1, dropnan=False):
    df_to_shift = df.copy()
    df_to_shift.index = df.date
#   df.head()
    agg = df.copy()
    agg.columns = [original+'(t)' for original in agg.columns]
    agg.rename(columns = {'site(t)':'site','date(t)':'date'},inplace = True)
    for i in range(n_in, 0, -1): 
        
        shifted = df_to_shift.shift(freq = '%dM'%i)
#        shifted.date = shifted.index ##updating new date
        shifted.drop(['percent', 'date'],inplace = True, axis = 1) # removing lagged fmc
        for col in shifted.columns:
            if col not in ['date','site']:
                shifted.rename(columns = {col: col+'(t-%d)'%i}, inplace = True)
        shifted.head()
        agg = pd.merge(agg,shifted, on = ['site','date'], how = 'left')
        agg.head()
    if dropnan:
        agg.dropna(inplace=True)
    return agg
#
#%%  
###############################################################################
RELOADINPUT = True
INPUTNAME = 'lstm_input_data_pure+all_same_07_may_2019'
LAG = 4

EPOCHS = int(2e4)
BATCHSIZE = 2048
DROPOUT = 0.1
LOAD_MODEL = True
SAVENAME = 'quality_pure+all_same_07_may_2019_small_v2'
OVERWRITE = True
RETRAIN = True

RETRAINEPOCHS = int(1e3)
###############################################################################

#%%modeling

if RELOADINPUT:
    dataset_with_nans = pd.read_pickle(INPUTNAME)
else:
    if os.path.isfile(INPUTNAME):
        raise  Exception('[INFO] Input File already exists. Try different INPUTNAME')
    dataset_with_nans = make_df()    
    dataset_with_nans.to_pickle(INPUTNAME)
    
##apply min max scaling
dataset = dataset_with_nans.dropna()
# integer encode forest cover
encoder = LabelEncoder()
dataset['forest_cover'] = encoder.fit_transform(dataset['forest_cover'].values)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset.drop(['site','date'],axis = 1).values)
rescaled = dataset.copy()
rescaled.loc[:,dataset.drop(['site','date'],axis = 1).columns] = scaled
reframed = series_to_supervised(rescaled, LAG,  dropnan = True)
reframed.reset_index(drop = True, inplace = True)
#print(reframed.head())
 
# split into train and test sets
train = reframed.loc[reframed.date.dt.year<2018].drop(['site','date'], axis = 1)
test = reframed.loc[reframed.date.dt.year>=2018].drop(['site','date'], axis = 1)
#print(train.shape)
#print(test.shape)
# split into input and outputs
train_X, train_y = train.drop(['percent(t)'], axis = 1).values, train['percent(t)'].values
test_X, test_y = test.drop(['percent(t)'], axis = 1).values, test['percent(t)'].values
# reshape input to be 3D [samples, timesteps, features]
train_Xr = train_X.reshape((train_X.shape[0], LAG+1, len(all_inputs)))
test_Xr = test_X.reshape((test_X.shape[0], LAG+1, len(all_inputs)))
#print(train_Xr.shape, train_y.shape, test_Xr.shape, test_y.shape)
 
#%% design network

filepath = os.path.join(dir_codes, 'model_checkpoint/LSTM/%s.hdf5'%SAVENAME)


def build_model(input_shape=(train_Xr.shape[1], train_Xr.shape[2])):
    
    model = Sequential()
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT, return_sequences=True))
    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT))
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True))
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT, return_sequences=True))
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True))   
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True))   
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True))
#    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT))
    #model.add(LSTM(50, input_shape=(train_Xr.shape[1], train_Xr.shape[2]), dropout = 0.3))
    #model.add(LSTM(10, input_shape=(train_Xr.shape[1], train_Xr.shape[2]), dropout = 0.3))
#    model.add(Dense(6))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    return model
    
checkpoint = ModelCheckpoint(filepath, save_best_only=True)
callbacks_list = [checkpoint]

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
#    rmse_diff, model_rmse = infer_importance(model,train_Xr, train_y,test_Xr, test_y,\
#         batch_size = BATCHSIZE, retrain_epochs = RETRAINEPOCHS, \
#         retrain = True, iterations = 10)        
#    rmse_diff.to_pickle(os.path.join(dir_codes, 'model_checkpoint/rmse_diff_%s'%save_name))
    #%% plot history
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='dev')
#    ax.set_ylim(0.004,0.02)
    ax.legend()
    ax.set_xlabel('epochs')
    ax.set_ylabel('mse')
    plt.show()
#%% 
#Predictions
yhat = model.predict(test_Xr)
#test_X = test_X.reshape((test_X.shape[0], (LAG+1)*test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:,-len(all_inputs):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
pred_frame = test.copy()
pred_frame.iloc[:,:len(all_inputs)+1] = scaler.inverse_transform(test.iloc[:,:len(all_inputs)+1])
pred_frame = pred_frame.iloc[:,:len(all_inputs)+1]
pred_frame = pred_frame.join(reframed.loc[:,['site','date']])
inv_y = pred_frame['percent(t)']
pred_frame['percent(t)_hat'] = inv_yhat
# calculate RMSE
rmse = sqrt(mean_squared_error(pred_frame['percent(t)'],pred_frame['percent(t)_hat'] ))


#%% true vsersus pred scatter
sns.set(font_scale=1.5, style = 'ticks')
plot_pred_actual(inv_y.values, inv_yhat, r2_score(inv_y, inv_yhat), rmse, ms = 30,\
                         zoom = 1.,dpi = 200,axis_lim = [0,300], xlabel = "FMC", mec = 'grey', mew = 0)

#%% persistence error
current = pred_frame.loc[:,['percent(t)','site','date']]
current.index = current.date
previous = current.shift(freq = '1M')
previous.date = previous.index
previous.reset_index(drop = True, inplace = True)
current.reset_index(drop = True, inplace = True)
both = pd.merge(current,previous, on = ['site','date'], how = 'left')
both.dropna(inplace = True)
persistence_rmse = sqrt(mean_squared_error(both['percent(t)_x'], both['percent(t)_y']))
print('[INFO] RMSE: %.3f' % rmse) 
print('[INFO] Persistence RMSE: %.3f' % persistence_rmse) 
print('[INFO] FMC Standard deviation : %.3f' % pred_frame['percent(t)'].std())

#%% plot predicted timeseries

train_frame = train.copy()
train_frame.iloc[:,:len(all_inputs)+1] = scaler.inverse_transform(train.iloc[:,:len(all_inputs)+1])
train_frame = train_frame.iloc[:,:len(all_inputs)+1]
train_frame = train_frame.join(reframed.loc[:,['site','date']])

frame = train_frame.append(pred_frame, sort = True)
#
for site in pred_frame.site.unique():
    sub = frame.loc[frame.site==site]
#    print(sub.shape)
    sub.index = sub.date
    if sub['percent(t)_hat'].count()<7:
        continue
    fig, ax = plt.subplots(figsize = (6,2))
    sub.plot(y = 'percent(t)', linestyle = '', markeredgecolor = 'grey', ax = ax,\
             marker = 'o', label = 'actual', color = 'grey')
    sub.plot(y = 'percent(t)_hat', linestyle = '', markeredgecolor = 'grey', ax = ax,\
             marker = 'o', label = 'predicted',color = 'fuchsia', ms = 8)
    ax.legend(loc = 'lower center',bbox_to_anchor=[0.5, -0.7], ncol=2)
    ax.set_ylabel('FMC(%)')
    ax.set_xlabel('')
    ax.axvspan(sub.index.min(), '2018-01-01', alpha=0.1, color='grey')
    ax.axvspan( '2018-01-01',sub.index.max(), alpha=0.1, color='fuchsia')
    ax.set_title(site)
    plt.show()
#%% sensitivity

def infer_importance(model, train_Xr, train_y, test_Xr, test_y, pred_frame,
                     retrain = True, iterations =1, retrain_epochs = int(1e3),\
                     batch_size = int(2e12)):
    pred_y = model.predict(test_Xr).flatten()
    model_rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    rmse_diff = pd.DataFrame(index = pred_frame.drop(['percent(t)','percent(t)_hat','site','drop'], axis = 1).columns,\
                             columns = range(iterations))
    for itr in range(iterations):
        for feature in rmse_diff.index:
            if retrain:
                sample_train_x = train_Xr.copy()
                sample_train_x.loc[:,feature] = 0.
                model.fit(sample_train_x.astype(float),train_y, epochs=retrain_epochs, batch_size=batch_size, \
                          verbose = False)
            sample_test_x = test_Xr.copy()
            sample_test_x.loc[:,feature] = 0.
            sample_pred_y = model.predict(sample_test_x).flatten()
            sample_rmse = np.sqrt(mean_squared_error(test_y, sample_pred_y))
            rmse_diff.loc[feature, itr] = sample_rmse - model_rmse
    rmse_diff['mean'] = rmse_diff.mean(axis = 'columns')
    rmse_diff['sd'] = rmse_diff.drop('mean',axis = 'columns').std(axis = 'columns')
    rmse_diff.drop(range(iterations),axis = 'columns', inplace = True)
    rmse_diff.dropna(subset = ['mean'], inplace = True, axis = 0)
#    print(rmse_diff)
    return rmse_diff, model_rmse

#%% data availability bar plot across features
    

#sns.set(font_scale=0.9, style = 'ticks')    
#dataset = dataset_with_nans.dropna(subset = ['percent'])
#fig, ax = plt.subplots(figsize = (2,6))
#(dataset.count()/dataset.shape[0]).sort_values().plot.barh(ax = ax, color = 'k')
#ax.set_xlabel('Fraction valid')
#  
###bar plot by site for vv availability
#vv_avail = (dataset.groupby('site').vv.count()/dataset.groupby('site').\
#            percent.count()).sort_values()
#sns.set(font_scale=0.7, style = 'ticks')    
#fig, ax = plt.subplots(figsize = (2,6))
#vv_avail.plot.barh(ax = ax, color = 'k')
#ax.set_xlabel('Fraction valid')  
#
#### map of SAR availability
#latlon = pd.read_csv(dir_data+"/fuel_moisture/nfmd_queried_latlon.csv", index_col = 0)
#latlon = latlon.join(pd.DataFrame(vv_avail, columns = ["vv_avail"]), how = 'right')
#latlon.dropna(inplace = True)
#
#fig, ax, m = plot_usa()
#plot=m.scatter(latlon.Longitude.values, latlon.Latitude.values, 
#                   s=30,c=latlon.vv_avail.values,cmap ='viridis' ,edgecolor = 'k',\
#                        marker='o',latlon = True, zorder = 2,\
#                        vmin = 0, vmax = 1)
#plt.setp(ax.spines.values(), color='w')
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.08)
#fig.colorbar(plot,ax=ax,cax=cax)
#ax.set_title('Fraction of times $\sigma_{VV}$ is available')
#plt.show()
#
#
#### site record length versus vv availability
#latlon = pd.read_csv(dir_data+"/fuel_moisture/nfmd_queried_latlon.csv", index_col = 0)
#latlon = latlon.join(pd.DataFrame(vv_avail, columns = ["vv_avail"]), how = 'right')
#latlon = latlon.join(pd.DataFrame(frame.groupby('site').date.min().\
#                                  rename('min_date')), how = 'right')
#fig, ax = plt.subplots(figsize = (3,3))
#ax.scatter(latlon.min_date.values,latlon.vv_avail.values, s = 40, \
#                       edgecolor = 'grey', lw = 1)
#ax.set_xlabel('Earliest recorded FMC at site')
#ax.set_ylabel('Site spceific $\sigma_{VV}$ availability')
#plt.xticks(rotation=40)

#%%individual sites error

### sites which have less training data
site_train_length = pd.DataFrame(train_frame.groupby('site').site.count().rename('train_length'))
site_rmse = pd.DataFrame(pred_frame.groupby('site').apply(lambda df: sqrt(mean_squared_error(\
                  df['percent(t)'], df['percent(t)_hat']))), columns = ['site_rmse'])
site_rmse = site_rmse.join(site_train_length)

fig, ax = plt.subplots()
site_rmse.plot.scatter(x = 'train_length',y='site_rmse', ax = ax, s = 40, \
                       edgecolor = 'grey', lw = 1)
ax.set_xlabel('No. of training examples available per site')
ax.set_ylabel('Site specific RMSE')
sns.set(font_scale=0.9, style = 'ticks')
site = 'TPEC_Presidio_TX'
alpha = 0.3
#for site in ['TPEC_Presidio_TX', 'CAF Truchas 1' ,'Panter','Los Alamos',\
#             'ponderosa','Los Robles, Thousand Oaks','San Marcos','Reyes Creek']:
for site in frame.site.unique():
    site_df = frame.loc[frame.site==site]
    if site_df.shape[0]<7:
        continue
    site_df.index = site_df.date
    fig, ax = plt.subplots(figsize = (4,2))
    site_df.loc[:,['percent(t)']].plot(marker = 'o', ax = ax, color = 'steelblue')
    site_df.loc[:,['percent(t)_hat']].plot(marker = 'o', ax = ax, \
                   color = 'steelblue', mec = 'orange')
    ax2 = ax.twinx()
    site_df.loc[:,['green(t)']].plot(marker = 'o', ax = ax2, color = 'g', alpha = alpha)
    ax3 = ax.twinx()
    site_df.loc[:,['vv(t)']].plot(marker = 'o', ax = ax3, color = 'm', alpha = alpha)
    
    ax.set_title(site)
    ax.legend(bbox_to_anchor = [1.2,1], loc = 'upper left')
    ax2.legend(bbox_to_anchor = [1.2,0.55], loc = 'center left')
    ax3.legend(bbox_to_anchor = [1.2,0.4], loc = 'center left')
    plt.show()
    