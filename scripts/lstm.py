# -*- coding: utf-8 -*-
"""
Created on SOct 20 2021

@author: kkrao
"""

import os
import sys
import pickle 
from math import sqrt
from numpy import concatenate
from scipy import optimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.offsets import MonthEnd, SemiMonthEnd

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, optimizers
import matplotlib.pyplot as plt

from init import dir_data, dir_codes, lc_dict
os.chdir(os.path.join(dir_codes,'scripts'))
from QC_of_sites import clean_fmc
from fnn_smoothed_anomaly_all_sites import plot_pred_actual, plot_importance, plot_usa
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable



def interpolate(df, var = 'percent', ts_start='2015-01-01', ts_end='2019-05-31', \
                resolution = '1M',window = '1M', max_gap = '4M'):
    """
    Interpolates data within df. This is used for interpolating LSTM data 
    because RNN needs LSTM at regular intervals, but LSTM is not measured
    at regular intervals in NFMD.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    var : TYPE, optional
        DESCRIPTION. The default is 'percent'.
    ts_start : TYPE, optional
        DESCRIPTION. The default is '2015-01-01'.
    ts_end : TYPE, optional
        DESCRIPTION. The default is '2019-05-31'.
    resolution : TYPE, optional
        DESCRIPTION. The default is '1M'.
    window : TYPE, optional
        DESCRIPTION. The default is '1M'.
    max_gap : TYPE, optional
        DESCRIPTION. The default is '4M'.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = df.copy()
    df.index = df.date
    df = df.resample(resolution).mean()


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

    return  df

def reindex(df, resolution = '1M'):
    """
    This is al alternate to interpolation. Rather than filling the gaps
    with bilinear method, this does by nerest neighbour (copies the LSTM of
     the nearest measured label)

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    resolution : TYPE, optional
        DESCRIPTION. The default is '1M'.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
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


def make_df(inputs, quality = 'pure+all same',resolution = 'SM',\
            max_gap = '3M', lag = '3M'):
    """
    Make input training dataframe from raw LFMC measurements downloaded 
    from NFMD. Dataframe has features and ground-truths.

    Parameters
    ----------
    inputs : TYPE
        DESCRIPTION.
    quality : TYPE, optional
        DESCRIPTION. The default is 'pure+all same'.
    resolution : TYPE, optional
        DESCRIPTION. The default is 'SM'.
    max_gap : TYPE, optional
        DESCRIPTION. The default is '3M'.
    lag : TYPE, optional
        DESCRIPTION. The default is '3M'.

  
    Returns
    -------
    new : TYPE
        DESCRIPTION.
    int_lag : TYPE
        DESCRIPTION.

    """
    ####FMC
    df = pd.read_pickle('fmc_24_may_2019')
    df = clean_fmc(df, quality = quality)
    master = pd.DataFrame()
    no_inputs_sites = []
    for site in df.site.unique():
        df_sub = df.loc[df.site==site]
        df_sub = reindex(df_sub,resolution = resolution)
        # no interpolation for LFMC meas. Just reindex to closest 15th day.
        # df_sub = interpolate(df_sub, 'percent', resolution = resolution, max_gap = max_gap)
        master = master.append(df_sub, ignore_index = True, sort = False)
    ## static inputs    
    static_features_all = pd.read_csv('static_features.csv',dtype = {'site':str}, index_col = 0)
    if not(static_inputs is None):
        static_features_subset = static_features_all.loc[:,static_inputs]
        master = master.join(static_features_subset, on = 'site') 
    ### optical
    
    df = pd.read_pickle('landsat8_500m_cloudless')

    opt = pd.DataFrame()
    uncer = pd.DataFrame()
    for site in master.site.unique():
        if site in df.site.values:
            df_sub = df.loc[df.site==site]  
            feature_sub = interpolate(df_sub, var = optical_inputs, resolution = resolution, max_gap = max_gap)
            feature_sub['site'] = site
            opt = opt.append(feature_sub, ignore_index = True, sort = False)
            ###calc incertainty introduced by interpolating 
            df_sub = reindex(df_sub,resolution = resolution)
            row = ((df_sub[optical_inputs] - feature_sub[optical_inputs])).abs().mean()
            row.name = site
            row['n'] = df_sub.shape[0]
            uncer = uncer.append(row)
            
        else:
            if site not in no_inputs_sites:
                print('[INFO]\tsite skipped :\t%s'%site)
                no_inputs_sites.append(site)
        # master = pd.merge(master,feature, on=['date','site'], how = 'outer')         
    uncer.to_csv(os.path.join(dir_data,'optical_uncertainty.csv'))
    sum_diffs = uncer.drop('n', axis = 1).multiply(uncer.loc[:,'n'], axis = 0)
    print(sum_diffs.sum()/uncer['n'].sum())
    ## 6.7% with respect to max range
    ### sar
    df = pd.read_pickle('sar_ascending_30_apr_2019')
    # for var in microwave_inputs:
    micro = pd.DataFrame()
    uncer = pd.DataFrame()
    for site in master.site.unique():
        if site in df.site.values:
            df_sub = df.loc[df.site==site]  
            feature_sub = interpolate(df_sub, var = microwave_inputs, resolution = resolution, max_gap = max_gap)
            feature_sub['site'] = site
            micro = micro.append(feature_sub, ignore_index = True, sort = False)
            
            ###calc incertainty introduced by interpolating 
            df_sub = reindex(df_sub,resolution = resolution)
            row = ((df_sub[microwave_inputs] - feature_sub[microwave_inputs])).abs().mean()
            row.name = site
            row['n'] = df_sub.shape[0]
            uncer = uncer.append(row)
        else:
            if site not in no_inputs_sites:
                print('[INFO]\tsite skipped :\t%s'%site)
                no_inputs_sites.append(site)
        # master = pd.merge(master,feature, on=['date','site'], how = 'outer')          
    uncer.to_csv(os.path.join(dir_data,'microwave_uncertainty.csv'))    
    sum_diffs = uncer.drop('n', axis = 1).multiply(uncer.loc[:,'n'], axis = 0)
    print(sum_diffs.sum()/uncer['n'].sum())
    # 3.0% with respect to max range
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


#%%modeling


def split_train_test(dataset, inputs = None, int_lag = None, CV = False, fold = None, FOLDS = 3, DROPCROPS = False, TRAINRATIO = 0.7):
    """
    Split data into training and testing

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    inputs : TYPE, optional
        DESCRIPTION. The default is None.
    int_lag : TYPE, optional
        DESCRIPTION. The default is None.
    CV : TYPE, optional
        DESCRIPTION. The default is False.
    fold : TYPE, optional
        DESCRIPTION. The default is None.
    FOLDS : TYPE, optional
        DESCRIPTION. The default is 3.
    DROPCROPS : TYPE, optional
        DESCRIPTION. The default is False.
    TRAINRATIO : TYPE, optional
        DESCRIPTION. The default is 0.7.

    Returns
    -------
    dataset : TYPE
        DESCRIPTION.
    rescaled : TYPE
        DESCRIPTION.
    reframed : TYPE
        DESCRIPTION.
    train_Xr : TYPE
        DESCRIPTION.
    test_Xr : TYPE
        DESCRIPTION.
    train_y : TYPE
        DESCRIPTION.
    test_y : TYPE
        DESCRIPTION.
    train : TYPE
        DESCRIPTION.
    test : TYPE
        DESCRIPTION.
    test_X : TYPE
        DESCRIPTION.
    scaler : TYPE
        DESCRIPTION.
    encoder : TYPE
        DESCRIPTION.

    """

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
    reframed = rescaled.copy()

    print('[INFO] Dataset has %d sites'%len(reframed.site.unique()))
    ####    
    reframed.reset_index(drop = True, inplace = True)
    #### split train test as 70% of time series of each site rather than blanket 2018 cutoff
    train_ind=[]
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

    train = reframed.loc[train_ind].drop(['site','date'], axis = 1)
    test = reframed.loc[~reframed.index.isin(train_ind)].drop(['site','date'], axis = 1)
    train.sort_index(inplace = True)
    test.sort_index(inplace = True)

    train_X, train_y = train.drop(['percent(t)'], axis = 1).values, train['percent(t)'].values
    test_X, test_y = test.drop(['percent(t)'], axis = 1).values, test['percent(t)'].values
    if inputs==None: #checksum
        inputs = all_inputs
    train_Xr = train_X.reshape((train_X.shape[0], int_lag+1, len(inputs)), order = 'A')
    test_Xr = test_X.reshape((test_X.shape[0], int_lag+1, len(inputs)), order = 'A')
    return dataset, rescaled, reframed, \
            train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
            scaler, encoder         

    


 
#%% design network




def build_model(input_shape, filepath):
    """Define keras model.
    """
    Areg = regularizers.l2(1e-5)
    Breg = regularizers.l2(1e-3)
    Kreg = regularizers.l2(1e-10)
    Rreg = regularizers.l2(1e-15)
    
    model = Sequential()
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                  return_sequences=True, \
                  bias_regularizer= Breg))
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                    return_sequences=True, \
                    bias_regularizer= Breg))
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                   bias_regularizer= Breg))
    model.add(Dense(1))
    model.compile(loss=LOSS, optimizer='Nadam')
    return model


#%% 
#Predictions
def predict(model, test_Xr, test_X, test, reframed, scaler, inputs):
    """
    Predict LFMC with a trained model on the test set.

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    test_Xr : TYPE
        DESCRIPTION.
    test_X : TYPE
        DESCRIPTION.
    test : TYPE
        DESCRIPTION.
    reframed : TYPE
        DESCRIPTION.
    scaler : TYPE
        DESCRIPTION.
    inputs : TYPE
        DESCRIPTION.

    Returns
    -------
    inv_y : TYPE
        DESCRIPTION.
    inv_yhat : TYPE
        DESCRIPTION.
    pred_frame : TYPE
        DESCRIPTION.
    rmse : TYPE
        DESCRIPTION.
    r2 : TYPE
        DESCRIPTION.

    """
    
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

def main():
    """
    Bring everything together. 
    Allows you to create input data to a LSTM, train it, or predict LFMC 
    using it.

    Returns
    -------
    None.

    """
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
    RETRAINEPOCHS = int(5e3)
    FOLDS = 3
    CV = False
    
    kf = KFold(n_splits=FOLDS, random_state = SEED)
    int_lag = int(LAG[0])
    
    # os.chdir(dir_data)
    # convert series to supervised learning
    pd.set_option('display.max_columns', 10)
    if RESOLUTION =='SM':
        int_lag*=2
        
        
        
    filepath = os.path.join(dir_codes,'trained_model/%s.hdf5'%SAVENAME)
        
    microwave_inputs = ['vv','vh']
    optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv']
    #optical_inputs = ['red','green','blue','swir','nir', 'ndvi', 'ndwi','nirv','vari','ndii']
    mixed_inputs =  ['vv_%s'%den for den in optical_inputs] + ['vh_%s'%den for den in optical_inputs] + ['vh_vv']
    dynamic_inputs = microwave_inputs + optical_inputs + mixed_inputs
    static_inputs = ['slope', 'elevation', 'canopy_height','forest_cover',\
                        'silt', 'sand', 'clay']
    
    all_inputs = static_inputs+dynamic_inputs
    inputs = all_inputs
        
    if RELOADINPUT:
        dataset= pd.read_pickle(os.path.join(dir_data, INPUTNAME))
    else:
        if os.path.isfile(INPUTNAME) and not(OVERWRITEINPUT):
            raise  Exception('[INFO] Input File already exists. Try different INPUTNAME')
        dataset, int_lag = make_df(resolution = RESOLUTION, max_gap = MAXGAP, lag = LAG, inputs = inputs)    
        dataset.to_pickle(INPUTNAME)
        
    dataset, rescaled, reframed, \
        train_Xr, test_Xr,train_y, test_y, train, test, test_X, \
        scaler, encoder = split_train_test(dataset, int_lag = int_lag, \
                                           FOLDS = FOLDS, DROPCROPS = DROPCROPS,\
                                               TRAINRATIO = TRAINRATIO, inputs = inputs)
        
    
    
    
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
        model = build_model(input_shape=(train_Xr.shape[1], train_Xr.shape[2]),\
                        filepath = filepath)
    
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
        
        
    inv_y, inv_yhat, pred_frame, rmse, r2  = predict(model, test_Xr, test_X, test, reframed, scaler, inputs)
    #%% true vsersus pred scatter
    sns.set(font_scale=1.5, style = 'ticks')
    plot_pred_actual(inv_y.values, inv_yhat,  np.corrcoef(inv_y.values, inv_yhat)[0,1]**2, rmse, ms = 30,\
                zoom = 1.,dpi = 200,axis_lim = [0,300], xlabel = "Actual LFMC", \
                ylabel = "Predicted LFMC",mec = 'grey', mew = 0, test_r2 = False, bias = True)

if __name__=="__main__":
    main()