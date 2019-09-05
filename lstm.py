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
os.chdir(dir_codes)
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

def interpolate(df, var = 'percent', ts_start='2015-01-01', ts_end='2019-05-31', \
                resolution = '1M',window = '1M', max_gap = '4M'):
    df = df.copy()
    df.index = df.date
    df = df.resample(resolution).mean()

        
    # df.dropna(subset = [var], inplace = True)
    # x_ = df.groupby(df.index).mean().resample('1d').asfreq().index.values
    # y_ = df.groupby(df.index).mean().resample('1d').asfreq()[var].interpolate().rolling(window = window).mean()
    # z = pd.Series(y_,x_)
    # df.sort_index(inplace = True)
    if resolution == '1M':
        interp_limit = int(max_gap[:-1])
    elif resolution =='SM':
        interp_limit = 2*int(max_gap[:-1])
    else:
        raise  Exception('[INFO] RESOLUTION not supported')
    df = df.interpolate(limit = interp_limit)    
    df = df.dropna()
    df = df[var]
    df['date'] = df.index
    # df['delta'] = df.index.to_series().diff()
    # gap_end = df.loc[df.delta>=max_gap].index
    # gap_start = df.loc[(df.delta>=max_gap).shift(-1).fillna(False)].index
    # for start, end in zip(gap_start, gap_end):
    #     z.loc[start:end] = np.nan
    # z =z.reindex(pd.date_range(start=ts_start, end=ts_end, freq=resolution))
    # z = pd.DataFrame(z)
    # z.dropna(inplace = True)
    # z['site'] = df.site[0]
    # z['date'] = z.index
    return  df

def reindex(df, resolution = '1M'):
    site = df.site.values[0]
    # if resolution == '1M':
    #     df.date = pd.to_datetime(df['date'], format="%Y%m") + MonthEnd(1)
    # elif resolution =='SM':
    #     df.date = df.date + SemiMonthEnd(1)
    # else:
    #     raise  Exception('[INFO] RESOLUTION not supported')
    # df = df.groupby('date').mean()

    df.index = df.date
    df = df.resample(rule = resolution, label = 'right' ).mean().dropna()
    df['date'] = df.index
    df['site'] = site
    return df

os.chdir(dir_data)
# convert series to supervised learning


microwave_inputs = ['vv','vh']
optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv']
mixed_inputs =  ['vv_%s'%den for den in optical_inputs] + ['vh_%s'%den for den in optical_inputs] + ['vh_vv']
dynamic_inputs = microwave_inputs + optical_inputs + mixed_inputs

static_inputs = ['slope', 'elevation', 'canopy_height','forest_cover',\
                    'silt', 'sand', 'clay']

all_inputs = static_inputs+dynamic_inputs
inputs = all_inputs
def make_df(quality = 'pure+all same',resolution = 'SM', max_gap = '3M', lag = '3M', inputs = inputs):
    ####FMC
    df = pd.read_pickle('fmc_24_may_2019')
    df = clean_fmc(df, quality = quality)
    master = pd.DataFrame()
    no_inputs_sites = []
    for site in df.site.unique():
        df_sub = df.loc[df.site==site]
        df_sub = reindex(df_sub,resolution = resolution)
        # df_sub = interpolate(df_sub, 'percent', resolution = resolution, max_gap = max_gap)
        master = master.append(df_sub, ignore_index = True, sort = False)
    ## static inputs    
    static_features_all = pd.read_csv('static_features.csv',dtype = {'site':str}, index_col = 0)
    if not(static_inputs is None):
        static_features_subset = static_features_all.loc[:,static_inputs]
        master = master.join(static_features_subset, on = 'site') 
    ### optical
    
    df = pd.read_pickle('landsat8_500m_cloudless')
    # for var in optical_inputs:
    opt = pd.DataFrame()
    for site in master.site.unique():
        if site in df.site.values:
            df_sub = df.loc[df.site==site]  
            feature_sub = interpolate(df_sub, var = optical_inputs, resolution = resolution, max_gap = max_gap)
            feature_sub['site'] = site
            opt = opt.append(feature_sub, ignore_index = True, sort = False)
        else:
            if site not in no_inputs_sites:
                print('[INFO]\tsite skipped :\t%s'%site)
                no_inputs_sites.append(site)
        # master = pd.merge(master,feature, on=['date','site'], how = 'outer')         
    ### sar
    df = pd.read_pickle('sar_ascending_30_apr_2019')
    # for var in microwave_inputs:
    micro = pd.DataFrame()
    for site in master.site.unique():
        if site in df.site.values:
            df_sub = df.loc[df.site==site]  
            feature_sub = interpolate(df_sub, var = microwave_inputs, resolution = resolution, max_gap = max_gap)
            feature_sub['site'] = site
            micro = micro.append(feature_sub, ignore_index = True, sort = False)
        else:
            if site not in no_inputs_sites:
                print('[INFO]\tsite skipped :\t%s'%site)
                no_inputs_sites.append(site)
        # master = pd.merge(master,feature, on=['date','site'], how = 'outer')          
        
    dyn = pd.merge(opt,micro, on=['date','site'], how = 'outer')
    ## micro/opt inputs
    for num in microwave_inputs:
        for den in optical_inputs:
            dyn['%s_%s'%(num, den)] = dyn[num]/dyn[den]
    dyn['vh_vv'] = dyn['vh']-dyn['vv']
    
    dyn = dyn[dynamic_inputs+['date','site']]
    dyn = dyn.dropna()

    ## start filling master
    
    if resolution == '1M':
        int_lag = int(lag[:-1])
    elif resolution =='SM':
        int_lag = 2*int(lag[:-1])
    else:
        raise  Exception('[INFO] RESOLUTION not supported')
        
        
    ##serieal    
    new = master.copy()
    new.columns = master.columns+'(t)'
    
    for i in range(int_lag, -1, -1):
        for col in list(dyn.columns):
            if col not in ['date','site']:
                if i==0:
                    new[col+'(t)'] = np.nan
                else:
                    new[col+'(t-%d)'%i] = np.nan
    for i in range(int_lag, 0, -1):
        for col in list(master.columns):
            if col not in ['date','site','percent']:
                new[col+'(t-%d)'%i] = new[col+'(t)']           
    new = new.rename(columns = {'date(t)':'date','site(t)':'site'})
    count=0        
    for index, row in master.iterrows():
        dyn_sub = dyn.loc[dyn.site==row.site]
        dyn_sub['delta'] = row.date - dyn_sub.date
        if resolution == '1M':
            dyn_sub['steps'] = (dyn_sub['delta'] /np.timedelta64(30, 'D')).astype('int')
        elif resolution =='SM':
            dyn_sub['steps'] = (dyn_sub['delta']/np.timedelta64(15, 'D')).astype('int')
        if all(elem in dyn_sub['steps'].values for elem in range(int_lag, -1, -1)):
            count+=1
            # break debugging
            # print('[INFO] %d'%count)
            dyn_sub = dyn_sub.loc[dyn_sub.steps.isin(range(int_lag, -1, -1))]
            dyn_sub = dyn_sub.sort_values('steps')
            # flat = dyn_sub.stack()
            # flat.index.get_level_values(level=1)
            flat = dyn_sub.pivot_table(columns = 'steps').T.stack().reset_index()
            flat['level_1'] = flat["level_1"].astype(str) + '(t-' +flat["steps"].astype(str)+ ')'
            flat.index = flat['level_1']
            flat.index = flat.index.str.replace('t-0', 't', regex=True)
            flat =  flat.drop(['steps', 'level_1'], axis = 1)[0]
            
            new.loc[index,flat.index] = flat.values
            # print('[INFO] Finding Observation to match measurement... %0.0f %% complete'%(index/new.shape[0]*100))
        sys.stdout.write('\r'+'[INFO] Finding Observation to match measurement... %0.0f %% complete'%(index/new.shape[0]*100))
        sys.stdout.flush()
    new = new.dropna()
    return new , int_lag       
    # count=0
    # site_count = 0
    # ##vectorized:
    # for site in master.site.unique():
    #     master_sub = master.loc[master.site==site]
    #     if site in dyn.site.values:
    #         dyn_sub = dyn.loc[dyn.site==site]
    #         delta = master_sub.date.values.astype('datetime64[D]') - dyn_sub.date.values.astype('datetime64[D]').reshape((len(dyn_sub.date), 1))
    #         if resolution == '1M':
    #             delta = (delta/np.timedelta64(30, 'D')).astype('int')
    #         elif resolution =='SM':
    #             delta = (delta/np.timedelta64(15, 'D')).astype('int')
    #         matches = np.repeat(0, delta.shape[1])
    #         for item in range(int_lag, -1, -1):
    #             matches+=(delta==item).any(axis = 0)
    #         count+=(matches==(int_lag+1)).sum() 
    #         if (matches==int_lag+1).sum() >=1:
    #             site_count+=1
    # print('[INFO] Examples = %d'%count)    
    # print('[INFO] Sites = %d'%site_count)
    
    
    
    # master.reset_index(drop = True, inplace = True)
    # return master    
        


# convert series to supervised learning
def series_to_supervised(df, n_in=1, dropnan=False):
    df_to_shift = df.copy()
    df_to_shift.index = df.date
#   df.head()
    agg = df.copy()
    agg.columns = [original+'(t)' for original in agg.columns]
    agg.rename(columns = {'site(t)':'site','date(t)':'date'},inplace = True)
    for i in range(n_in, 0, -1): 
        
        shifted = df_to_shift.shift(freq = '%dd'%(i*15))
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



##input options 
SEED = 0
np.random.seed(SEED)
RELOADINPUT = True
OVERWRITEINPUT = False
LOAD_MODEL = True
OVERWRITE = False
RETRAIN = False
SAVEFIG = False
DROPCROPS = True
RESOLUTION = 'SM'
MAXGAP = '3M'
INPUTNAME = 'lstm_input_data_pure+all_same_28_may_2019_res_%s_gap_%s'%(RESOLUTION, MAXGAP)
SAVENAME = 'quality_pure+all_same_28_may_2019_res_%s_gap_%s_site_split'%(RESOLUTION, MAXGAP)

##modeling options
EPOCHS = int(20e3)
BATCHSIZE = int(2e5)
DROPOUT = 0.05
TRAINRATIO = 0.70
LOSS = 'mse'
LAG = '3M'
RETRAINEPOCHS = int(20e3)
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
Kreg = regularizers.l2(1e-15)
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
#    optim = optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)
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

#%% true versus pred site means
t = pred_frame.groupby('site')['percent(t)','percent(t)_hat'].mean()
x = t['percent(t)'].values
y = t['percent(t)_hat'].values

plot_pred_actual(x, y,\
        np.corrcoef(x, y)[0,1]**2, \
        sqrt(mean_squared_error(x, y)), \
        ms = 40,\
        zoom = 1.,dpi = 200,axis_lim = [50,200], xlabel = "Actual site-averaged LFMC", \
        ylabel = "Predicted site-averaged LFMC",mec = 'grey', mew = 2,test_r2 = False)

# #%% true versus pred seasonality

# t = pred_frame.groupby(['site',pred_frame.date.dt.month])['percent(t)','percent(t)_hat'].mean()
# x = t['percent(t)'].values
# y = t['percent(t)_hat'].values

# plot_pred_actual(x, y,\
#         r2_score(x, y), \
#         sqrt(mean_squared_error(x, y)), \
#         ms = 40,\
#         zoom = 1.,dpi = 200,axis_lim = [50,200], xlabel = "Actual MoY-averaged FMC", \
#         ylabel = "Predicted MoY-averaged FMC",mec = 'grey', mew = 1)


# #%% true versus pred IAV

# t = pred_frame.groupby(['site',pred_frame.date.dt.year])['percent(t)','percent(t)_hat'].mean()
# x = t['percent(t)'].values
# y = t['percent(t)_hat'].values

# plot_pred_actual(x, y,\
#         r2_score(x, y), \
#         sqrt(mean_squared_error(x, y)), \
#         ms = 40,\
#         zoom = 1.,dpi = 200,axis_lim = [50,200], xlabel = "Actual annual FMC", \
#         ylabel = "Predicted annual FMC",mec = 'grey', mew = 1)

#%% split predict plot into pure and mixed species sites

#### plot only pure species point
#
#sites = pd.read_excel('fuel_moisture/NFMD_sites_QC.xls', index_col = 0)
#pure_species_sites = sites.loc[(sites.include==1)&(sites.comment.isin(['only 1'])),'site']
#x = pred_frame.loc[pred_frame.site.isin(pure_species_sites),'percent(t)'].values
#y = pred_frame.loc[pred_frame.site.isin(pure_species_sites),'percent(t)_hat'].values
#plot_pred_actual(x, y,\
#        r2_score(x, y), \
#        sqrt(mean_squared_error(x, y)), \
#        ms = 30,zoom = 1.,dpi = 200,axis_lim = [0,300], \
#        xlabel = "FMC", mec = 'grey', mew = 0)
#
#####plot mixed species points as well
#mixed_species_sites = sites.loc[(sites.include==1)&(sites.comment.isin(['all same'])),'site']
#x = pred_frame.loc[pred_frame.site.isin(mixed_species_sites),'percent(t)'].values
#y = pred_frame.loc[pred_frame.site.isin(mixed_species_sites),'percent(t)_hat'].values
#plot_pred_actual(x, y,\
#        r2_score(x, y), \
#        sqrt(mean_squared_error(x, y)), \
#        ms = 30,zoom = 1.,dpi = 200,axis_lim = [0,300], \
#        xlabel = "FMC", mec = 'grey', mew = 0)
#

#%% persistence error
#current = pred_frame.loc[:,['percent(t)','site','date']]
#current.index = current.date
#previous = current.shift(freq = '1M')
#previous.date = previous.index
#previous.reset_index(drop = True, inplace = True)
#current.reset_index(drop = True, inplace = True)
#both = pd.merge(current,previous, on = ['site','date'], how = 'left')
#both.dropna(inplace = True)
#persistence_rmse = sqrt(mean_squared_error(both['percent(t)_x'], both['percent(t)_y']))

# print('[INFO] RMSE: %.3f' % rmse) 
#print('[INFO] Persistence RMSE: %.3f' % persistence_rmse) 
# print('[INFO] FMC Standard deviation : %.3f' % pred_frame['percent(t)'].std())

#%%individual sites error

## sites which have less training data

train_frame = train.copy()
train_frame.iloc[:,:] = scaler.inverse_transform(train.iloc[:,:])
# train_frame = train_frame.iloc[:,:len(inputs)+1]
train_frame = train_frame.join(reframed.loc[:,['site','date']])
frame = train_frame.append(pred_frame, sort = True)

site_train_length = pd.DataFrame(train_frame.groupby('site').site.count().rename('train_length'))
site_rmse = pd.DataFrame(pred_frame.groupby('site').apply(lambda df: sqrt(mean_squared_error(\
                  df['percent(t)'], df['percent(t)_hat']))), columns = ['site_rmse'])
site_rmse = site_rmse.join(frame.groupby('site')['percent(t)'].std().rename('norm_site_rmse'))
site_rmse = site_rmse.join(site_train_length)
site_rmse['norm_site_rmse'] = site_rmse['site_rmse']/site_rmse['norm_site_rmse']

site_rmse = site_rmse.sort_values(by = 'site_rmse', ascending = False)
# fig, ax = plt.subplots()
# site_rmse.plot.scatter(x = 'train_length',y='site_rmse', ax = ax, s = 40, \
#                        edgecolor = 'grey', lw = 1)
# ax.set_xlabel('No. of training examples available per site')
# ax.set_ylabel('Site specific RMSE')

# fig, ax = plt.subplots()
# site_rmse.plot.scatter(x = 'train_length',y='norm_site_rmse', ax = ax, s = 40, \
#                        edgecolor = 'grey', lw = 1)
# ax.set_xlabel('No. of training examples available per site')
# ax.set_ylabel('Site specific NRMSE')

#%% ignoring very poor sites
low_rmse_sites = site_rmse.loc[site_rmse.site_rmse<=40].index

x = pred_frame.loc[pred_frame.site.isin(low_rmse_sites),'percent(t)'].values
y = pred_frame.loc[pred_frame.site.isin(low_rmse_sites),'percent(t)_hat'].values
# plot_pred_actual(x, y,\
#         r2_score(x, y), \
#         sqrt(mean_squared_error(x, y)), \
#         ms = 30,zoom = 1.,dpi = 200,axis_lim = [0,300], \
#         xlabel = "FMC", mec = 'grey', mew = 0)

high_rmse_sites = list(set(site_rmse.index) - set(low_rmse_sites))

#%% plot predicted timeseries


#
#sns.set(font_scale=0.9, style = 'ticks')  
#for site in pred_frame.site.unique():
#    sub = frame.loc[frame.site==site]
##    print(sub.shape)
#    sub.index = sub.date
##    if sub['percent(t)_hat'].count()<7:
##        continue
#    fig, ax = plt.subplots(figsize = (4,1.5))
#    sub.plot(y = 'percent(t)', linestyle = '', markeredgecolor = 'grey', ax = ax,\
#             marker = 'o', label = 'actual', color = 'None', mew =2)
#    sub.plot(y = 'percent(t)_hat', linestyle = '', markeredgecolor = 'fuchsia', ax = ax,\
#             marker = 'o', label = 'predicted',color = 'None', mew= 2)
#    ax.legend(loc = 'lower center',bbox_to_anchor=[0.5, -0.7], ncol=2)
#    ax.set_ylabel('FMC(%)')
#    ax.set_xlabel('')
#    ax.axvspan(sub.index.min(), '2018-01-01', alpha=0.1, color='grey')
#    ax.axvspan( '2018-01-01',sub.index.max(), alpha=0.1, color='fuchsia')
#    ax.set_title(site)
#
#    if SAVEFIG:
#        os.chdir(dir_codes)
#        fig.savefig('plots/%s.jpg'%site, bbox_inches='tight')
#    plt.show()

# sns.set(font_scale=0.9, style = 'ticks')  
# alpha = 0.2
# for site in high_rmse_sites:
#     sub = frame.loc[frame.site==site]
# #    print(sub.shape)
#     sub.index = sub.date
# #    if sub['percent(t)_hat'].count()<7:
# #        continue
#     fig, ax = plt.subplots(figsize = (4,1.5))
#     l1 = ax.plot(sub.index, sub['percent(t)'], linestyle = '-',\
#             zorder = 99, markeredgecolor = 'grey',\
#             marker = 'o', label = 'actual FMC', color = 'None', mew =2)
#     l2 = ax.plot(sub.index, sub['percent(t)_hat'], linestyle = '-', \
#             zorder = 100, markeredgecolor = 'fuchsia', \
#             marker = 'o', label = 'predicted FMC',color = 'None', mew= 2)
#     ax.set_ylabel('FMC(%)')
#     ax.set_xlabel('')
    
#     ax2 = ax.twinx()
#     l3 = ax2.plot(sub.index, sub['green(t)'], ms = 5, mew = 0,alpha = alpha, \
#                     marker = 'o', label = 'green', color = 'g')
#     ax3 = ax.twinx()
#     l4 = ax3.plot(sub.index, sub['vv(t)'], ms = 5, mew = 0,alpha = alpha,\
#                     marker = 'o', label = 'vv',color = 'orange')    
#     ax.set_title(site)
#     ls = l1+l2+l3+l4
#     labs = [l.get_label() for l in ls]
#     ax.tick_params(axis='x', rotation=45)
#     ax.legend(ls, labs, loc = 'lower center',bbox_to_anchor=[0.5, -1],\
#               ncol=2)
#     plt.show()

#%% sensitivity
os.chdir(dir_data)
def infer_importance(rmse, r2, iterations =1, retrain_epochs = RETRAINEPOCHS,\
                      batch_size = BATCHSIZE):

    rmse_diff = pd.DataFrame(index = ['microwave','optical'],\
                              columns = range(iterations))
    r2_diff = pd.DataFrame(index = ['microwave','optical'],\
                              columns = range(iterations))
    for itr in range(iterations):
        for feature_set in rmse_diff.index.values:
            print('[INFO] Fitting model for %s inputs only'%feature_set)
            if feature_set =='microwave':
                inputs = list(set(all_inputs)-set(optical_inputs))
                _, _, reframed, \
                train_Xr, test_Xr,train_y, test_y, _, test, test_X, \
                scaler,_ = \
                split_train_test(dataset, \
                                  inputs = inputs )
            elif feature_set=='optical':           
                inputs = list(set(all_inputs)-set(microwave_inputs)-set(mixed_inputs))
                _, _, reframed, \
                train_Xr, test_Xr,train_y, test_y, _, test, test_X, \
                scaler,_ = \
                split_train_test(dataset, \
                      inputs = inputs)
            model = build_model(input_shape=(train_Xr.shape[1], train_Xr.shape[2]))

            history = model.fit(train_Xr, train_y, epochs=retrain_epochs, \
                                batch_size=batch_size,\
                    validation_data=(test_Xr, test_y), verbose=0, shuffle=False)
            _,_,_, sample_rmse, sample_r2  = predict(model, test_Xr, test_X,\
                                          test, reframed, scaler, inputs)
            
            rmse_diff.loc[feature_set, itr] = sample_rmse - rmse
            r2_diff.loc[feature_set, itr] = sample_r2 - r2
    rmse_diff['mean'] = rmse_diff.mean(axis = 'columns')
    rmse_diff['sd'] = rmse_diff.drop('mean',axis = 'columns').std(axis = 'columns')
    rmse_diff.drop(range(iterations),axis = 'columns', inplace = True)
    rmse_diff.dropna(subset = ['mean'], inplace = True, axis = 0)
    
    r2_diff['mean'] = r2_diff.mean(axis = 'columns')
    r2_diff['sd'] = r2_diff.drop('mean',axis = 'columns').std(axis = 'columns')
    r2_diff.drop(range(iterations),axis = 'columns', inplace = True)
    r2_diff.dropna(subset = ['mean'], inplace = True, axis = 0)
#    print(rmse_diff)
    return rmse_diff, r2_diff

# rmse_diff, r2_diff = infer_importance(rmse, r2,  retrain_epochs = RETRAINEPOCHS,iterations = 1)
# print(rmse_diff)
# print(r2_diff)
#    rmse_diff.to_pickle(os.path.join(dir_codes, 'model_checkpoint/rmse_diff_%s'%save_name))
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


#%% seasonal cycle rmsd
os.chdir(dir_data)
df = pd.read_pickle('fmc_24_may_2019')
df.date = pd.to_datetime(df.date)
df.loc[df.percent>=1000,'percent'] = np.nan
df = df.loc[~df.fuel.isin(['1-Hour','10-Hour','100-Hour', '1000-Hour',\
                        'Duff (DC)', '1-hour','10-hour','100-hour',\
                        '1000-hour', 'Moss, Dead (DMC)' ])]
df = df[df.date.dt.year>=2014]
## res = 1M
# seasonal_mean = pd.DataFrame(index = range(1, 13))
# for site in df.site.unique():
#     df_sub = df.loc[df.site==site]
#     # seasonality_site = interpolate(df_sub, ts_start = df_sub.date.min())
#     seasonality_site = df_sub.groupby(df_sub.date.dt.month).percent.mean().rename(site)
#     seasonal_mean = seasonal_mean.join(seasonality_site)
# seasonal_mean.to_pickle('seasonal_mean_all_sites_1M_31_may_2019')

# res = 'SM' 
# seasonal_mean = pd.DataFrame(index = range(1, 25))
# df['moy'] = df.date.dt.month # add fortnite or year index
# df['foy'] = 2*df['moy'] - 1*(df.date.dt.day<15) # < because 15th and later are counted as month end in 'SM'
# for site in df.site.unique():
#     df_sub = df.loc[df.site==site]
#     # seasonality_site = interpolate(df_sub, ts_start = df_sub.date.min())
#     seasonality_site = df_sub.groupby('foy').percent.mean().rename(site)
#     counts = pd.value_counts(df_sub['foy'])
#     seasonality_site = seasonality_site[counts>5]
#     if seasonality_site.shape[0]==0:
#         continue
#     seasonal_mean = seasonal_mean.join(seasonality_site)
# seasonal_mean.to_pickle('seasonal_mean_all_sites_SM_31_may_2019')


seasonal_mean = pd.read_pickle('seasonal_mean_all_sites_%s_31_may_2019'%RESOLUTION)
frame['1M'] = frame.date.dt.month
frame['SM'] = (2*frame['1M'] - 1*(frame.date.dt.day<=15)).astype(int)
# <= because <15 is replaced with 15 in pandas SM
frame['percent_seasonal_mean'] = np.nan
for site in frame.site.unique():
    df_sub = frame.loc[frame.site==site,['site','date',RESOLUTION,'percent(t)']]
    if site not in seasonal_mean.columns:
        continue
    df_sub = df_sub.join(seasonal_mean.loc[:,site].rename('percent_seasonal_mean'), on = RESOLUTION)
    frame.update(df_sub)
    # frame.loc[frame.site==site,['site','date','mod','percent(t)','percent_seasonal_mean']]
frame.dropna(inplace = True, subset = ['percent_seasonal_mean'])

### manually calculating because if there is a site with only 1 observation per MoY, its rmsd will be zero
#there are 97 such sites
gg = frame['percent(t)_hat']-frame['percent_seasonal_mean']
gg = gg[gg.abs()>0]
sqrt((gg**2).mean())

rmsd = sqrt(mean_squared_error(frame['percent(t)'],frame['percent_seasonal_mean'] ))
print('[INFO] RMSD between actual and seasonal cycle: %.3f' % rmsd) 
print('[INFO] RMSE: %.3f' % rmse) 
#print('[INFO] Persistence RMSE: %.3f' % persistence_rmse) 
print('[INFO] FMC Standard deviation : %.3f' % frame['percent(t)'].std())


#%% true versus pred seasonal anomaly
ndf = frame[['site','date','percent(t)','percent(t)_hat','percent_seasonal_mean']]
ndf.dropna(inplace = True)

x = ndf.groupby(['site']).apply(lambda x: (x['percent(t)'] - x['percent(t)'].mean())).values
y = ndf.groupby(['site']).apply(lambda x: (x['percent(t)_hat'] - x['percent(t)'].mean())).values

# x = ndf['percent(t)'].values
# y = ndf['percent(t)_hat'].values

plot_pred_actual(x, y,\
        np.corrcoef(x, y)[0,1]**2, \
        sqrt(mean_squared_error(x, y)), \
        ms = 40,\
        zoom = 1.,dpi = 200,axis_lim = [-100,100],xlabel = "Actual LFMC anomaly", \
        ylabel = "Predicted LFMC anomaly",mec = 'None', mew = 1, test_r2 = False, cmap = "plasma")


#%% CV 
CV = False # hard coded
if CV:
    frame = pd.DataFrame()
    # model = build_model()
    for fold in range(FOLDS):
        print('[INFO] Fold: %1d'%fold)
        dataset, rescaled, reframed, \
            train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
            scaler, encoder = split_train_test(dataset, int_lag = int_lag, CV = CV, fold = fold)
            
        filepath = os.path.join(dir_codes, 'model_checkpoint/LSTM/%s_fold_%d.hdf5'%(SAVENAME, fold))
        
        checkpoint = ModelCheckpoint(filepath, save_best_only=True)
    
        callbacks_list = [checkpoint, earlystopping]
    
        history = model.fit(train_Xr, train_y, epochs=EPOCHS, batch_size=BATCHSIZE,\
                            validation_data=(test_Xr, test_y), verbose=0, shuffle=False,\
                            callbacks=callbacks_list)
        model = load_model(filepath) # once trained, load best model
        inv_y, inv_yhat, pred_frame, rmse, r2  = predict(model, test_Xr, test_X, test, reframed, scaler, inputs)
        frame = frame.append(pred_frame)
    x = frame['percent(t)'].values
    y =  frame['percent(t)_hat'].values
    rmse = np.sqrt(mean_squared_error(x,y))
    plot_pred_actual(x, y,  np.corrcoef(x, y)[0,1]**2, rmse, ms = 30,\
            zoom = 1.,dpi = 200,axis_lim = [0,300], xlabel = "Actual LFMC", \
            ylabel = "Predicted LFMC",mec = 'grey', mew = 0, test_r2 = False, bias = True)

    frame.to_csv(os.path.join(dir_data,'model_predictions_all_sites.csv'))

#%% RMSE vs sites. bar chart

frame = pd.read_csv(os.path.join(dir_data,'model_predictions_all_sites.csv'))
rmse = frame.groupby('site').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat']))).sort_values()
rmse.index = range(len(rmse))

fig, ax = plt.subplots(figsize = (6,6))
ax.bar(rmse.index, rmse.values, width = 1)
ax.set_ylabel('RMSE')
ax.set_xlabel('Sites')
# ax.set_xticklabels(range(len(rmse)))

#%% timeseries for three sites
new_frame = frame.copy()
new_frame.index = pd.to_datetime(new_frame.date)
rmse = new_frame.groupby('site').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat']))).sort_values()

fig, ax = plt.subplots(figsize = (4,1.5))
sub = new_frame.loc[new_frame.site == rmse.index[0]]
sub.plot(y = 'percent(t)', linestyle = '-', markeredgecolor = 'grey', ax = ax,\
        marker = 'o', label = 'actual', color = 'grey', mew =0.1,ms = 3,linewidth = 1 ,legend = False)
sub.plot(y = 'percent(t)_hat', linestyle = '-', markeredgecolor = 'fuchsia', ax = ax,\
        marker = 'o', label = 'predicted',color = 'None', mew= 0.1, ms = 3, lw = 1, legend = False)
ax.set_ylabel('FMC(%)')
ax.set_xlabel('')
# ax.set_title(site)


#%% performance by landcover table

table = pd.DataFrame({'RMSE':frame.groupby('forest_cover(t)').apply(lambda df: np.sqrt(mean_squared_error(df['percent(t)'],df['percent(t)_hat'])))})
table['R2'] = frame.groupby('forest_cover(t)').apply(lambda df: np.corrcoef(df['percent(t)'],df['percent(t)_hat'])[0,1]**2)
table['N'] = frame.groupby('forest_cover(t)').apply(lambda df: df.shape[0])
table['Bias'] = frame.groupby('forest_cover(t)').apply(lambda df: (df['percent(t)'] - df['percent(t)_hat']).mean())
### works only with original encoder!!
pkl_file = open('encoder.pkl', 'rb')
encoder = pickle.load(pkl_file) 
pkl_file.close()
table.index = encoder.inverse_transform(table.index.astype(int))
table.index = table.index.map(lc_dict)
table.index.name = 'landcover'
print(table)
table.to_excel('model_performance_by_lc.xls')


#%%


#x = pred_frame['percent(t)']
#y = pred_frame['percent_seasonal_mean']
#plot_pred_actual(x.values, y.values, r2_score(x, y), rmsd, ms = 30,\
#                 zoom = 1.,dpi = 200,axis_lim = [0,300], xlabel = "FMC", mec = 'grey', mew = 0)
#
##    
####rmsd by site
#for site in pred_frame.site.unique():
#    df_sub = pred_frame.loc[pred_frame.site==site,['site','date','mod','percent(t)','percent_seasonal_mean']]
#    rmsd = sqrt(mean_squared_error(df_sub['percent(t)'],df_sub['percent_seasonal_mean']))
#    print('[INFO] RMSD for site \t {}= \t {:.2f}'.format(site, rmsd))
#### choose some high rmsd site and check calculation
#site = 'D11_Miller_Gulch'
#df_sub = pred_frame.loc[pred_frame.site==site,['site','date','mod','percent(t)','percent_seasonal_mean']]
#df_sub
#seasonal_mean[site]
#to_plot = df.loc[df.site==site,['date','percent']]
#to_plot.index = to_plot.date
#to_plot.percent.plot()
#to_plot.index.min()





#%% Histogram of forest cover in the study area

# sns.set(font_scale=1, style = 'ticks') 
# hist = pd.value_counts(encoder.inverse_transform(dataset.drop_duplicates('site')['forest_cover(t)']), normalize = True)
# hist.index = hist.index.to_series().map(lc_dict)
# hist = hist.sort_index()
# fig, ax = plt.subplots(figsize = (4,4))
# hist.plot(kind = 'bar', ax = ax)
# ax.set_ylabel('No. of sites')

#%%
# site based heatmap
# rank = pred_frame.groupby('site').mean().drop(['percent(t)_hat'],axis = 1)
# for col in rank.columns:
#     if col[-3]=='-':
#         rank.drop(col, axis = 1, inplace = True)
# rank.columns = rank.columns.str[:-3]
# rank = rank.loc[site_rmse.sort_values('site_rmse', ascending = False).index]
# rank = rank.join(site_rmse)
# # rank = rank.join(reframed.groupby('site').count()['percent(t)'].rename('examples'))
# rank = rank.join(reframed.groupby('site').std()['percent(t)'].rename('fmc_sd'))
# # rank = rank.join(reframed.groupby('site').max()['percent(t)'].rename('fmc_max'))
# # rank = rank.join((reframed.groupby('site').max()-reframed.groupby('site').min())['percent(t)'].rename('fmc_range'))
# rank = rank.rename(columns = {'percent':'fmc'})
# sns.set(font_scale=0.5, style = 'ticks')  
# axs= sns.clustermap(rank.astype(float), standard_scale =1, row_cluster=False, col_cluster = True,  figsize = (8,6))

#%% seasonality vs. IAV

# score = pred_frame.groupby('site').mean()
# score['seasonality_score'] = np.nan
# score['IAV_score'] = np.nan
# score = score[['seasonality_score','IAV_score']]
# ##check seasonality
# def seasonality_score(df):
#     if df.shape[0]<=4:
#         return np.nan
#     else:
#         score = np.corrcoef(df['percent(t)'],df['percent(t)_hat'])[0,1]
#     return score
# ##check IAV
# def IAV_score(df):
#     # df =df.dropna()
#     mean = df.mean()
#     ### asumes all predictions span one year at max
#     score = (mean['percent(t)_hat'] - mean['percent(t)'])/mean['percent(t)']
#     return score

# for site in score.index:
#     sub = pred_frame.loc[pred_frame.site==site]
#     score.loc[site,'seasonality_score'] = seasonality_score(sub)
#     score.loc[site,'IAV_score'] = IAV_score(sub)
# score.dropna(inplace = True)
# sns.set(font_scale=1.4, style = 'ticks')
# fig, ax = plt.subplots(figsize = (4,4))
# ax.set_xlim(-1,1)
# ax.set_ylim(-1,1)
# sns.kdeplot(score.seasonality_score, score.IAV_score, ax = ax, cmap = 'Blues', shade = True, shade_lowest=False)
# ax.scatter(score.seasonality_score, score.IAV_score, marker = '+', color = 'k', linewidth = 0.5, s = 15)
# plt.show()
# print(score.shape)

#%% weights for mixed species


# def solve_for_weights(true_sub):
#     no_of_weights = len(true_sub.fuel.unique())
#         # data provided
#     x0 = np.repeat(0.,no_of_weights)
#     bounds=((0, 1),)*no_of_weights
#     X = true_sub.pivot(values = 'percent',columns = 'fuel')
#     X = X.fillna(X.mean()) ### important assumption. missing fmc replaced by mean
# #    X.dropna(inplace = True)
#     #sort columns alphabatically so that weight can be mapped to species later
#     X = X.reindex(sorted(X.columns), axis=1)
#     target = true_sub.groupby(true_sub.index)['percent(t)_hat'].mean()
#     target = target.loc[X.index]

#     def loss(x):
#         return mean_squared_error(target,X.dot(x))**0.5
#     cons = {'type':'eq',
#             'fun':lambda x: 1 - np.sum(x)}
    
#     opt = {'disp':False}
    
#     res = optimize.minimize(loss, x0, constraints=cons,
#                                   method='SLSQP', options=opt, bounds = bounds)
#     ## return optimization result, target FMC (FMC predicted from model),
#     ## True measured FMC multiplied by weights, simply averaged FMC with weights = 1/n
    
#     return res, target, X.dot(res.x), X.mean(axis = 1)


# df, int_lag = make_df(quality = 'only mixed')
# df.to_pickle('mixed_species_inputs')

#%% uncomment for mixed species plotting
# df = pd.read_pickle('mixed_species_inputs')
# dataset, rescaled, reframed, \
#     train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
#     scaler, encoder = split_train_test(df, inputs = inputs, int_lag = int_lag)
    
# test_Xr = np.concatenate((train_Xr, test_Xr))
# test = train.append(test)
# test_X = test.drop(['percent(t)'], axis = 1).values
# inv_y, inv_yhat, pred_frame, rmse, r2  = predict(model, test_Xr, test_X, test, reframed, scaler, inputs)

#%%
# true = pd.read_pickle('fmc_04-29-2019')
# true.site = true.site.astype('str')
# true.date = pd.to_datetime(true.date)
# true.date = true.date + MonthEnd(1)

# true = true.loc[true.date.dt.year>=2015]
# true = true.loc[~true.fuel.isin(['1-Hour','10-Hour','100-Hour', '1000-Hour'])] 
# true = true[~true.fuel.str.contains("Duff")] ###dropping all dead DMC
# true = true[~true.fuel.str.contains("Dead")] ###dropping all dead DMC
# true = true[~true.fuel.str.contains("DMC")] ###dropping all dead DMC
# no_of_species_in_sites = true.groupby('site').fuel.unique().apply(lambda series: len(series))
# sites_with_leq_4_species = no_of_species_in_sites.loc[(no_of_species_in_sites<=4)].index
# true = true.loc[true.site.isin(sites_with_leq_4_species)]
# true.index = pd.to_datetime(true.date)
# ctr_found = 0
# ctr_notfound = 0
# ctr_more_than_1_fuels = 0


# W = pd.DataFrame(index = true.site.unique(), columns = ['W1','W2','W3','W4','RMSE'])
# W = W.sort_index()
# col_list = ['site','date','S1','S2','S3','S4','W1','W2','W3','W4','FMC_weighted','FMC_mean','FMC_hat','RMSE']
# optimized_df = pd.DataFrame(columns = col_list)


# for site in true.site.unique():
#     if site in pred_frame.site.values:
# #            print('[INFO] Site found')
#         ctr_found+=1
#         true_sub = true.loc[true.site==site,['site','date','fuel','percent']]
#         true_sub = true_sub.groupby(['date','fuel']).mean().reset_index()
#         # true_sub = true_sub.set_index('date')
        
#         pred_sub = pred_frame.loc[pred_frame.site==site,['date','percent(t)_hat']].copy()
#         true_sub = true_sub.set_index('date').join(pred_sub.set_index('date'), how = 'inner', on = 'date')
#         if true_sub.shape[0]<20:
#             continue
# #            print('[INFO] No. of fuels = %d'%len(true_sub.fuel.unique()))
#         if len(true_sub.fuel.unique())>1:
#             ctr_more_than_1_fuels+=1
#             res, FMC_hat, FMC_weighted, FMC_mean = solve_for_weights(true_sub)
#             w = res.x
#             rmse = res.fun
#             if len(w)<4:
#                 w = np.append(w, np.repeat(np.nan, 4-len(w)))
#             w = np.append(w, rmse)
#             W.loc[site] = w
#             ##### fill outputs of optimization in optimized df
#             FMC_hat.name = 'FMC_hat'
#             FMC_weighted.name = 'FMC_weighted'
#             FMC_mean.name = 'FMC_mean'
# #            break
#             _odf = pd.concat([FMC_hat, FMC_weighted, FMC_mean], axis=1)
#             _odf = _odf.reindex(columns = col_list)                
#             _odf['date'] = _odf.index
#             _odf['site'] = site
#             _odf.loc[:,['W1','W2','W3','W4','RMSE']] = w 
#             species = sorted(true_sub.fuel.unique())
#             if len(species)<4:
#                 species = species+['nan']*(4-len(species))
#             _odf.loc[:,['S1','S2','S3','S4']] = species
#             optimized_df = optimized_df.append(_odf, ignore_index = True)
#             print('[INFO]\tWeights computed for site\t%s'%site)
#     else:
#         ctr_notfound+=1
# #            print('[INFO] Site NOT found')

# W = W.infer_objects()       
# W.dropna(how = 'all', inplace = True)
# #W.to_pickle("mixed_species/mixed_species_weights")
# #optimized_df.to_pickle('mixed_species/optimization_results')     
    
   
# ####### hist of RMSE
# #sns.set(font_scale=1.5, style = 'ticks')
# #fig, ax = plt.subplots(figsize = (3,3))
# #W.hist(column = 'RMSE', ax = ax, bins = 50, density = True)
# #ax.set_xlabel('RMSE')
# #ax.set_ylabel('Probability Density')
# #ax.set_xlim(0,10)

# ############## pred FMC vs optimized FMC
# sns.set(font_scale=1.2, style = 'ticks')  
# ax = plot_pred_actual(optimized_df['FMC_hat'], optimized_df['FMC_weighted'], \
#                   r2_score(optimized_df['FMC_hat'], optimized_df['FMC_weighted']),\
#                 mean_squared_error(optimized_df['FMC_weighted'], optimized_df['FMC_hat'])**0.5,\
#                       zoom = 1,dpi = 150, axis_lim = [75,150], ms = 20,\
#                       xlabel = '$\hat{FMC}$', ylabel = '$\sum w_i\ x\ FMC_i$')    

#%%############## pred FMC vs mean FMC (simple averaged)
# x = inv_y
# y = inv_yhat
# plot_pred_actual(x.values, y, r2_score(x, y), mean_squared_error(x,y)**0.5, ms = 30,\
#                 zoom = 1.,dpi = 200,axis_lim = [0,300],mec = 'grey', mew = 0,\
#                     xlabel = '$\overline{FMC}$', ylabel = '$\hat{FMC}$')
#%%
# ax = plot_pred_actual(optimized_df['FMC_hat'],optimized_df['FMC_mean'], \
#                   r2_score(optimized_df['FMC_hat'], optimized_df['FMC_mean'] ),\
#                   mean_squared_error(optimized_df['FMC_mean'], optimized_df['FMC_hat'])**0.5,\
#                       zoom = 1,dpi = 150, axis_lim = [75,150], ms = 20, \
#                       ylabel = '$\overline{FMC}$', xlabel = '$\hat{FMC}$')

# ######## weights plot

# Wmax = optimized_df.groupby('site')[['W1','W2','W3','W4']].max().max(1)
# Wmin = optimized_df.groupby('site')[['W1','W2','W3','W4']].max().min(1)
# fig, ax = plt.subplots(figsize = (3,3))
# sns.kdeplot(Wmin, Wmax,cmap="Blues", shade=True, ax = ax, shade_lowest = False,n_levels=100)
# ax.scatter(Wmin, Wmax, edgecolor = 'k', s = 15, facecolor = 'b')
# ax.set_xlim(-0.1,1.1)
# ax.set_ylim(-0.1,1.1)
# ax.set_xlabel('min($w_i$)')
# ax.set_ylabel('max($w_i$)')
#######