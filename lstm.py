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
from scipy import optimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.offsets import MonthEnd


from sklearn.metrics import mean_squared_error, r2_score
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
            110:'shrub',
            120:'shrub',
            130:'shrub',
            140:'grass',
            150:'sparse vegetation',
            160:'regularly flooded forest'}

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
inputs = all_inputs
def make_df(quality = 'pure+all same'):
    ####FMC
    df = pd.read_pickle('fmc_04-29-2019')
    df = clean_fmc(df, quality = quality)
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
    master['vh_vv'] = master['vh']-master['vv']
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

INPUTNAME = 'lstm_input_data_pure+all_same_21_may_2019_vv_vh'
SAVENAME = 'quality_pure+all_same_7_may_2019_small_train_test_tailor_split'

##input options 
RELOADINPUT = True
LOAD_MODEL = True
OVERWRITE = False
RETRAIN = False
SAVEFIG = False
DROPCROPS = True
##modeling options
EPOCHS = int(20e3)
BATCHSIZE = 2048
DROPOUT = 0.15
TRAINRATIO = 0.7
LOSS = 'mse'
LAG = 4
RETRAINEPOCHS = int(20e3)

###############################################################################

#%%modeling

if RELOADINPUT:
    dataset_with_nans = pd.read_pickle(INPUTNAME)
else:
    if os.path.isfile(INPUTNAME):
        raise  Exception('[INFO] Input File already exists. Try different INPUTNAME')
    dataset_with_nans = make_df()    
    dataset_with_nans.to_pickle(INPUTNAME)
    
def split_train_test(dataset_with_nans,inputs = None):
    if inputs != None:
        dataset = dataset_with_nans.loc[:,['site','date', 'percent']+inputs].dropna()
    else:
        dataset = dataset_with_nans.dropna()
    if DROPCROPS:
        crop_classes = [item[0] for item in lc_dict.items() if item[1] == 'crop']
        dataset = dataset.loc[~dataset.forest_cover.isin(crop_classes)]
    # integer encode forest cover
    encoder = LabelEncoder()
    dataset['forest_cover'] = encoder.fit_transform(dataset['forest_cover'].values)
    # normalize features
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(dataset.drop(['site','date'],axis = 1).values)
    rescaled = dataset.copy()
    rescaled.loc[:,dataset.drop(['site','date'],axis = 1).columns] = scaled
    reframed = series_to_supervised(rescaled, LAG,  dropnan = True)
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
    for site in reframed.site.unique():
        sub = reframed.loc[reframed.site==site]
        sub = sub.sort_values(by = 'date')
        train_ind = train_ind+list(sub.index[:int(np.ceil(sub.shape[0]*TRAINRATIO))])
    train = reframed.loc[train_ind].drop(['site','date'], axis = 1)
    test = reframed.loc[~reframed.index.isin(train_ind)].drop(['site','date'], axis = 1)
    
    #print(train.shape)
    #print(test.shape)
    # split into input and outputs
    train_X, train_y = train.drop(['percent(t)'], axis = 1).values, train['percent(t)'].values
    test_X, test_y = test.drop(['percent(t)'], axis = 1).values, test['percent(t)'].values
    # reshape input to be 3D [samples, timesteps, features]
    if inputs==None: #checksum
        inputs = all_inputs
    train_Xr = train_X.reshape((train_X.shape[0], LAG+1, len(inputs)))
    test_Xr = test_X.reshape((test_X.shape[0], LAG+1, len(inputs)))
    return dataset, rescaled, reframed, \
            train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
            scaler, encoder
            
dataset, rescaled, reframed, \
    train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
    scaler, encoder = split_train_test(dataset_with_nans, inputs = inputs)

#print(train_Xr.shape, train_y.shape, test_Xr.shape, test_y.shape)
 
#%% design network

filepath = os.path.join(dir_codes, 'model_checkpoint/LSTM/%s.hdf5'%SAVENAME)

Areg = regularizers.l2(1e-4)
Breg = regularizers.l2(1e-4)
Kreg = regularizers.l2(1e-5)
Rreg = regularizers.l2(1e-5)

def build_model(input_shape=(train_Xr.shape[1], train_Xr.shape[2])):
    
    model = Sequential()
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                   return_sequences=True))#, \
#                   activity_regularizer = Areg, \
#                   bias_regularizer= Breg,\
#                   kernel_regularizer = Kreg, \
#                   recurrent_regularizer = Rreg))
    # model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT,return_sequences=True))
    model.add(LSTM(10, dropout = DROPOUT, recurrent_dropout=DROPOUT))#, \
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
#    model.add(Dense(6))
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
    # model = load_model(filepath) # once trained, load best model

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
    inv_yhat = concatenate((yhat, test_X[:,-len(inputs):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    pred_frame = test.copy()
    pred_frame.iloc[:,:len(inputs)+1] = scaler.inverse_transform(test.iloc[:,:len(inputs)+1])
    pred_frame = pred_frame.iloc[:,:len(inputs)+1]
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
plot_pred_actual(inv_y.values, inv_yhat, r2, rmse, ms = 30,\
                         zoom = 1.,dpi = 200,axis_lim = [0,300], xlabel = "FMC", mec = 'grey', mew = 0)

#
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
train_frame.iloc[:,:len(inputs)+1] = scaler.inverse_transform(train.iloc[:,:len(inputs)+1])
train_frame = train_frame.iloc[:,:len(inputs)+1]
train_frame = train_frame.join(reframed.loc[:,['site','date']])
frame = train_frame.append(pred_frame, sort = True)

site_train_length = pd.DataFrame(train_frame.groupby('site').site.count().rename('train_length'))
site_rmse = pd.DataFrame(pred_frame.groupby('site').apply(lambda df: sqrt(mean_squared_error(\
                  df['percent(t)'], df['percent(t)_hat']))), columns = ['site_rmse'])
site_rmse = site_rmse.join(frame.groupby('site')['percent(t)'].std().rename('norm_site_rmse'))
site_rmse = site_rmse.join(site_train_length)
site_rmse['norm_site_rmse'] = site_rmse['site_rmse']/site_rmse['norm_site_rmse']
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
                split_train_test(dataset_with_nans, \
                                 inputs = inputs )
            elif feature_set=='optical':           
                inputs = list(set(all_inputs)-set(microwave_inputs)-set(mixed_inputs))
                _, _, reframed, \
                train_Xr, test_Xr,train_y, test_y, _, test, test_X, \
                scaler,_ = \
                split_train_test(dataset_with_nans, \
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
#os.chdir(dir_data)
#df = pd.read_pickle('fmc_04-29-2019')
#df.date = pd.to_datetime(df.date)
#df.loc[df.percent>=1000,'percent'] = np.nan
#seasonal_mean = pd.DataFrame(index = range(1, 13))
#for site in df.site.unique():
#    df_sub = df.loc[df.site==site]
#    seasonality_site = interpolate(df_sub, ts_start = df_sub.date.min())
#    seasonality_site = seasonality_site.groupby(seasonality_site.date.dt.month).percent.mean().rename(site)
#    seasonal_mean = seasonal_mean.join(seasonality_site)
#seasonal_mean.to_pickle('seasonal_mean_all_sites')
    
seasonal_mean = pd.read_pickle('seasonal_mean_all_sites')
pred_frame['mod'] = pred_frame.date.dt.month
pred_frame['percent_seasonal_mean'] = np.nan
for site in pred_frame.site.unique():
    df_sub = pred_frame.loc[pred_frame.site==site,['site','date','mod','percent(t)']]
    df_sub = df_sub.join(seasonal_mean.loc[:,site].rename('percent_seasonal_mean'), on = 'mod')
    pred_frame.update(df_sub)
    pred_frame.loc[pred_frame.site==site,['site','date','mod','percent(t)','percent_seasonal_mean']]
rmsd = sqrt(mean_squared_error(pred_frame['percent(t)'],pred_frame['percent_seasonal_mean'] ))
print('[INFO] RMSD between actual and seasonal cycle: %.3f' % rmsd) 
print('[INFO] RMSE: %.3f' % rmse) 
#print('[INFO] Persistence RMSE: %.3f' % persistence_rmse) 
print('[INFO] FMC Standard deviation : %.3f' % pred_frame['percent(t)'].std())


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


# hist = pd.value_counts(encoder.inverse_transform(dataset.drop_duplicates('site')['forest_cover']))
# hist.index = hist.index.to_series().map(lc_dict)

# fig, ax = plt.subplots(figsize = (4,4))
# hist.plot(kind = 'bar', ax = ax)
# ax.set_ylabel('No. of sites')

#%%
## site based heatmap
# rank = pred_frame.groupby('site').mean().drop(['percent_seasonal_mean','mod','percent(t)_hat'],axis = 1)
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
# axs= sns.clustermap(rank.astype(float), standard_scale =1, row_cluster=False, col_cluster = True,  figsize = (6,4))

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


def solve_for_weights(true_sub):
    no_of_weights = len(true_sub.fuel.unique())
       # data provided
    x0 = np.repeat(0.,no_of_weights)
    bounds=((0, 1),)*no_of_weights
    X = true_sub.pivot(values = 'percent',columns = 'fuel')
    X = X.fillna(X.mean()) ### important assumption. missing fmc replaced by mean
#    X.dropna(inplace = True)
    #sort columns alphabatically so that weight can be mapped to species later
    X = X.reindex(sorted(X.columns), axis=1)
    target = true_sub.groupby(true_sub.index)['percent(t)_hat'].mean()
    target = target.loc[X.index]

    def loss(x):
        return mean_squared_error(target,X.dot(x))**0.5
    cons = {'type':'eq',
            'fun':lambda x: 1 - np.sum(x)}
    
    opt = {'disp':False}
    
    res = optimize.minimize(loss, x0, constraints=cons,
                                 method='SLSQP', options=opt, bounds = bounds)
    ## return optimization result, target FMC (FMC predicted from model),
    ## True measured FMC multiplied by weights, simply averaged FMC with weights = 1/n
    
    return res, target, X.dot(res.x), X.mean(axis = 1)


df = make_df(quality = 'only mixed')

dataset, rescaled, reframed, \
    train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
    scaler, encoder = split_train_test(df, inputs = inputs)
test_Xr = np.concatenate((train_Xr, test_Xr))
test = train.append(test)
test_X = test.drop(['percent(t)'], axis = 1).values
inv_y, inv_yhat, pred_frame, rmse, r2  = predict(model, test_Xr, test_X, test, reframed, scaler, inputs)


true = pd.read_pickle('fmc_04-29-2019')
true.site = true.site.astype('str')
true.date = pd.to_datetime(true.date)
true.date = true.date + MonthEnd(1)

true = true.loc[true.date.dt.year>=2015]
true = true.loc[~true.fuel.isin(['1-Hour','10-Hour','100-Hour', '1000-Hour'])] 
true = true[~true.fuel.str.contains("Duff")] ###dropping all dead DMC
true = true[~true.fuel.str.contains("Dead")] ###dropping all dead DMC
true = true[~true.fuel.str.contains("DMC")] ###dropping all dead DMC
no_of_species_in_sites = true.groupby('site').fuel.unique().apply(lambda series: len(series))
sites_with_leq_4_species = no_of_species_in_sites.loc[(no_of_species_in_sites<=4)].index
true = true.loc[true.site.isin(sites_with_leq_4_species)]
true.index = pd.to_datetime(true.date)
ctr_found = 0
ctr_notfound = 0
ctr_more_than_1_fuels = 0


W = pd.DataFrame(index = true.site.unique(), columns = ['W1','W2','W3','W4','RMSE'])
W = W.sort_index()
col_list = ['site','date','S1','S2','S3','S4','W1','W2','W3','W4','FMC_weighted','FMC_mean','FMC_hat','RMSE']
optimized_df = pd.DataFrame(columns = col_list)


for site in true.site.unique():
    if site in pred_frame.site.values:
#            print('[INFO] Site found')
        ctr_found+=1
        true_sub = true.loc[true.site==site,['site','date','fuel','percent']]
        true_sub = true_sub.groupby(['date','fuel']).mean().reset_index()
        # true_sub = true_sub.set_index('date')
        
        pred_sub = pred_frame.loc[pred_frame.site==site,['date','percent(t)_hat']].copy()
        true_sub = true_sub.set_index('date').join(pred_sub.set_index('date'), how = 'inner', on = 'date')
        if true_sub.shape[0]<20:
            continue
#            print('[INFO] No. of fuels = %d'%len(true_sub.fuel.unique()))
        if len(true_sub.fuel.unique())>1:
            ctr_more_than_1_fuels+=1
            res, FMC_hat, FMC_weighted, FMC_mean = solve_for_weights(true_sub)
            w = res.x
            rmse = res.fun
            if len(w)<4:
                w = np.append(w, np.repeat(np.nan, 4-len(w)))
            w = np.append(w, rmse)
            W.loc[site] = w
            ##### fill outputs of optimization in optimized df
            FMC_hat.name = 'FMC_hat'
            FMC_weighted.name = 'FMC_weighted'
            FMC_mean.name = 'FMC_mean'
#            break
            _odf = pd.concat([FMC_hat, FMC_weighted, FMC_mean], axis=1)
            _odf = _odf.reindex(columns = col_list)                
            _odf['date'] = _odf.index
            _odf['site'] = site
            _odf.loc[:,['W1','W2','W3','W4','RMSE']] = w 
            species = sorted(true_sub.fuel.unique())
            if len(species)<4:
                species = species+['nan']*(4-len(species))
            _odf.loc[:,['S1','S2','S3','S4']] = species
            optimized_df = optimized_df.append(_odf, ignore_index = True)
            print('[INFO]\tWeights computed for site\t%s'%site)
    else:
        ctr_notfound+=1
#            print('[INFO] Site NOT found')

W = W.infer_objects()       
W.dropna(how = 'all', inplace = True)
#W.to_pickle("mixed_species/mixed_species_weights")
#optimized_df.to_pickle('mixed_species/optimization_results')     
    
   
####### hist of RMSE
#sns.set(font_scale=1.5, style = 'ticks')
#fig, ax = plt.subplots(figsize = (3,3))
#W.hist(column = 'RMSE', ax = ax, bins = 50, density = True)
#ax.set_xlabel('RMSE')
#ax.set_ylabel('Probability Density')
#ax.set_xlim(0,10)
############## pred FMC vs optimized FMC
sns.set(font_scale=1.2, style = 'ticks')  
ax = plot_pred_actual(optimized_df['FMC_hat'], optimized_df['FMC_weighted'], \
                 r2_score(optimized_df['FMC_hat'], optimized_df['FMC_weighted']),\
                mean_squared_error(optimized_df['FMC_weighted'], optimized_df['FMC_hat'])**0.5,\
                     zoom = 1.5,dpi = 150, axis_lim = [75,150], ms = 20,\
                     xlabel = '$\hat{FMC}$', ylabel = '$\sum w_i\ x\ FMC_i$')    

############## pred FMC vs mean FMC (simple averaged)
ax = plot_pred_actual(optimized_df['FMC_hat'],optimized_df['FMC_mean'], \
                 r2_score(optimized_df['FMC_hat'], optimized_df['FMC_mean'] ),\
                 mean_squared_error(optimized_df['FMC_mean'], optimized_df['FMC_hat'])**0.5,\
                     zoom = 1.5,dpi = 150, axis_lim = [75,150], ms = 20, \
                     ylabel = '$\overline{FMC}$', xlabel = '$\hat{FMC}$')

######## weights plot

Wmax = optimized_df.groupby('site')[['W1','W2','W3','W4']].max().max(1)
Wmin = optimized_df.groupby('site')[['W1','W2','W3','W4']].max().min(1)
fig, ax = plt.subplots(figsize = (3,3))
sns.kdeplot(Wmin, Wmax,cmap="Blues", shade=True, ax = ax, shade_lowest = False,n_levels=100)
ax.scatter(Wmin, Wmax, edgecolor = 'k', s = 15, facecolor = 'b')
ax.set_xlim(-0.1,1.1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('min($w_i$)')
ax.set_ylabel('max($w_i$)')
#######