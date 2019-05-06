# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 18:59:11 2018

@author: kkrao
"""
## computation
import os
import numpy as np
import pandas as pd
from dirs import dir_data, dir_codes, dir_figures
##nn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import keras as k
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
### statistics
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
### plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import seaborn as sns
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap


def seasonal_anomaly(absolute,save = False, return_mean = False):
    """
    index should be datetime
    
    """
    mean = absolute.groupby(absolute.index.dayofyear).mean()   
    mean = mean.loc[absolute.index.dayofyear]
    mean.reset_index(drop = True, inplace = True)
    mean.index = absolute.index
    anomaly = absolute - mean
    anomaly.index.name = absolute.index.name+'_anomaly'
    if save:
        anomaly.to_pickle('cleaned_anomalies_11-29-2018/%s'%anomaly.index.name)
    if return_mean:
        return anomaly, mean
    else:
        return anomaly

def make_df(dynamic_features, static_features, response = 'fm_anomaly',\
            data_loc = 'cleaned_anomalies_11-29-2018/',sites=None):

    Df = pd.DataFrame()
    
    for feature in dynamic_features:
        df = pd.read_pickle('%s%s'%(data_loc,feature))
        if sites is None:
            sites = df.columns
        df = df.loc[:,sites].stack(dropna=False)
        df.name = feature
        Df = pd.concat([Df,df],axis = 'columns')
    Df.dropna(subset =[response],inplace = True)
    Df.fillna(0, inplace = True)
    Df.index.names= ['date','site']
    Df.reset_index(inplace=True)
    Df.site = Df.site.values
    static_features_all = pd.read_csv('static_features.csv',dtype = {'site':str}, index_col = 0)
    if not(static_features is None):
        static_features_subset = static_features_all.loc[:,static_features]
        Df = Df.join(static_features_subset, on = 'site')
    Df.reset_index(inplace=True, drop = True)
#    Df.drop(['site','date'],axis = 1, inplace = True)
    Df = Df.infer_objects()
#    Df.apply(pd.to_numeric, errors='ignore')
    return Df
    

def build_data(df, columns, response = 'fm_anomaly',ignore_multi_spec = False):
#    filter =
#        (df.residual.abs() <=delta_days)\
#        &(~df.Fuel.isin(['10-Hour', '100-Hour', '1000-Hour']))\
#        &(df.ndvi>=0.1)\
#        &(df.percent<=1000)\
#        &(df.percent>=40)\
#        &(df.residual_opt<=delta_days)
#    df = df.loc[filter,:]
#    if ignore_multi_spec:
#        df = ignore_multi_spec_fun(df, pct_thresh = pct_thresh)
#    columns = ['vv','percent','vh','red','green','blue','swir', 'nir',\
#               'ndvi', 'ndwi','nirv','slope', 'elevation', 'canopy_height',
#                'silt', 'sand', 'clay', 'incidence_angle', 'vh/ndvi',\
#               'latitude', 'longitude', 'vh/vv','vv/ndvi']
#    columns = ['vv_angle_corr','percent','vh_angle_corr','swir',\
#              'slope', 'elevation', 'canopy_height',
#                'silt', 'sand', 'clay', 'incidence_angle', 'vh/ndvi',\
#               'latitude', 'longitude', 'vh/vv','vv/ndvi']
#    columns = ['percent',\
#          'slope', 'elevation', 'canopy_height',
#            'silt', 'sand', 'clay',\
#           'latitude', 'longitude']
    n_features = len(columns)-1
    df = df.loc[:,columns]
    df.dropna(subset = [response],inplace = True)
    df.fillna(method = 'bfill', axis = 0,inplace = True)
    df.fillna(method = 'ffill', axis = 0, inplace = True)
#    df.fillna(0, inplace = True)
    dont_norm = df[response].copy()
#    df = (df - df.mean())/(df.std())    #normalize 
#    print(df.shape)    
    df.loc[:,response] = dont_norm
    train, test = train_test_split(df, train_size = 0.8, test_size = 0.2)
    train_y= train[response]
    train_x= train.drop([response],axis = 'columns')
    test_y= test[response]
    test_x= test.drop([response],axis = 'columns')
#    print(df.head())
    return train_x, train_y, test_x, test_y, n_features
    
def ignore_multi_spec_fun(df, pct_thresh = 20): 
    to_select = pd.DataFrame()
    for site in df.Site.unique():
        d = df.loc[df.Site==site,:]
        for date in d.meas_date.unique():
            x = d.loc[d.meas_date == date,:]
            if x.percent.max() - x.percent.min()<= pct_thresh:
                to_select = to_select.append(x)
    return to_select


def scheduler(epoch):
    lr_decay = 0.9
    if epoch%100 == 0:
        old_lr = K.get_value(model.optimizer.lr)
        new_lr = old_lr*lr_decay
        if new_lr<=1e-4:
            return  K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, new_lr)
    return K.get_value(model.optimizer.lr)

def build_model(n_features):
	# create model
    reg = regularizers.l1(1e-3)
    model = Sequential()
    model.add(Dense(15, input_dim = n_features,kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(150, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(15, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(60, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(30, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(30, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(15, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
    model.add(Dense(60, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
    model.add(Dense(30, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
    model.add(Dense(30, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
    model.add(Dense(15, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(6, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
    model.add(Dense(1, kernel_initializer='normal',kernel_regularizer=reg))
#    sgd = Adam(lr=1e-1)
#    sgd = SGD(lr=1e-2, momentum=0.9, decay=1e-4, nesterov=True)
    sgd = Adam(lr=3e-2, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])
    return model

def infer_importance(model, train_x, train_y, test_x, test_y, change_lr,\
                     retrain = False, iterations =1, retrain_epochs = int(1e3),\
                     batch_size = int(2e12)):
    pred_y = model.predict(test_x).flatten()
    model_rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    rmse_diff = pd.DataFrame(index = test_x.columns,\
                             columns = range(iterations))
    for itr in range(iterations):
        for feature in test_x.columns:
#        for feature in ['vv_anomaly', 'vh_anomaly', 'blue_anomaly', 'green_anomaly', 'red_anomaly', 'nir_anomaly', 'ndvi_anomaly', 'ndwi_anomaly', 'vv_ndvi_anomaly', 'vh_ndvi_anomaly']:
            if retrain:
                sample_train_x = train_x.copy()
                sample_train_x.loc[:,feature] = 0.
                model.fit(sample_train_x.astype(float),train_y, epochs=retrain_epochs, batch_size=batch_size, \
                          callbacks=[change_lr], verbose = False)
            sample_test_x = test_x.copy()
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

def infer_importance_by_var_category(model, train_x, train_y, test_x, test_y, change_lr,\
                     retrain = False, iterations =1, retrain_epochs = int(1e3),\
                     batch_size = int(2e12)):
    pred_y = model.predict(test_x).flatten()
    model_rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    categories = ['optical','microwave']
    rmse_diff = pd.DataFrame(index = categories,\
                             columns = range(iterations))
    
    micro=['vv','vv_angle_corr','vh_angle_corr','vh', 'vh/ndvi','vv/ndvi','vh/vv',\
           "vh_pm","vh_am","vv_pm","vv_am",\
             "vh_pm_ndvi", "vh_am_ndvi", "vv_pm_ndvi",\
             "vv_am_ndvi", 'vv_ndvi',\
             'vh_ndvi', 'vh_red', 'vv_red','vh_blue', 'vv_blue',\
             'vh_nir', 'vv_nir',\
             'vh_green', 'vv_green']
    opt=['red','green','blue','swir','nir',
         'ndvi', 'ndwi','nirv']
    micro = micro+[x+"_smoothed" for x in micro]+[x+"_anomaly" for x in micro]
    opt = opt+[x+"_smoothed" for x in opt]+[x+"_anomaly" for x in opt]
    for itr in range(iterations):
        for category in categories:
            if category=='optical':
                features = [x for x in opt if x in train_x.columns]
            else:
                features = [x for x in micro if x in train_x.columns]
                
            if retrain:
                sample_train_x = train_x.copy()
                sample_train_x.loc[:,features] = 0.
                model.fit(sample_train_x,train_y, epochs=retrain_epochs, batch_size=batch_size, \
                          callbacks=[change_lr], verbose = False)
            sample_test_x = test_x.copy()
            sample_test_x.loc[:,features] = 0.
            sample_pred_y = model.predict(sample_test_x).flatten()
            sample_rmse = np.sqrt(mean_squared_error(test_y, sample_pred_y))
            rmse_diff.loc[category, itr] = sample_rmse - model_rmse
            print('Model RMSE = %0.4f \t RMSE loss due to %s = %0.4f'%(model_rmse, category, rmse_diff.loc[category, itr]) )
    rmse_diff['mean'] = rmse_diff.mean(axis = 'columns')
    rmse_diff['sd'] = rmse_diff.drop('mean',axis = 'columns').std(axis = 'columns')
    rmse_diff.drop(range(iterations),axis = 'columns', inplace = True)
    rmse_diff.dropna(subset = ['mean'], inplace = True, axis = 0)
#    print(rmse_diff)
    return rmse_diff, model_rmse

def color_based_on_lc(fc):
    

    group_dict = {20: 0,
           30: 0,
           70: 1,
           100:2,
           110:3,
           120:3,
           130:3,
           140:4}
    groups = [group_dict[x] for x in fc.values]
    return groups
    
def plot_pred_actual(test_y, pred_y, R2, model_rmse,  cmap = 'plasma', axis_lim = [-25,50],\
                     xlabel = "FMC anomaly", zoom = 1,\
                     figname = None, dpi = 600,ms = 8):
    fig, ax = plt.subplots(figsize = (zoom*3.5,zoom*3.5), dpi = dpi)
    plt.axis('scaled')
    x = test_y
    y = pred_y
    #x=train_y
    #y = model.predict(train_x).flatten()
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
#    groups = color_based_on_lc(df.loc[test_x.index,'forest_cover'])
    plot = ax.scatter(x,y, c=z, s=ms, edgecolor='', cmap = cmap)
    
    ax.set_xlim(axis_lim)
    ax.set_ylim(axis_lim)
    ax.plot(axis_lim,axis_lim, lw =.5, color = 'grey')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel('Predicted '+xlabel)
    ax.set_xlabel('Actual '+xlabel)
    ax.set_yticks(ax.get_xticks())
#    ax.set_xticks([-50,0,50,100])
#    ax.set_yticks([-50,0,50,100])
    ax.annotate('$R^2_{test}=%0.2f$\n$RMSE=%0.1f$'%(np.floor(R2*100)/100, model_rmse), \
                xy=(0.05, 0.97), xycoords='axes fraction',\
                ha='left',va='top')
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.08)
#    fig.colorbar(plot,ax=ax,cax=cax)
#    ticklabels = groupwise_rmse(test_y, pred_y, groups)
#    cax.set_yticklabels(ticklabels.values())
##    cax.set_ylabel('Canopy height (m)')
    if not(figname is None):
        plt.savefig(os.path.join(dir_figures,figname), dpi = dpi,\
                    bbox_inches="tight")
    plt.show()
    
    return ax
    
def decompose_plot_pred_actual(pred_y, test_y, df, dpi = 600):
    pred_y_series = pd.Series(pred_y, index = test_y.index)
    pred_y_series.name = 'pred_y'
    
    temp_df = pd.DataFrame({'pred_y':pred_y, 'test_y':test_y})
    temp_df = temp_df.join(df.loc[:,['site','date']])
    mean = temp_df.copy()
    mean = mean.groupby('site').mean()
    anomaly = temp_df.copy()
    anomaly.index = temp_df.date
    fm = pd.read_pickle(os.path.join(dir_data, \
                     'cleaned_anomalies_11-29-2018/fm_smoothed'))
    _,fm = seasonal_anomaly(fm, return_mean = True)
    for site in anomaly.site.unique():
        sub = anomaly.loc[anomaly.site==site,:].copy()
        sub = sub.join(fm.loc[:,site])
        sub.pred_y-=sub[site]
        sub.test_y-=sub[site]
        anomaly.loc[anomaly.site==site,:]= sub.copy()
    
    
    rmse = np.sqrt(mean_squared_error(mean.test_y, mean.pred_y))
    r2 = r2_score(mean.test_y, mean.pred_y)
    plot_pred_actual(mean.test_y, mean.pred_y, r2, rmse, \
                     xlabel = "FMC average", axis_lim=[0,200], zoom = 1,\
                     cmap = sns.cubehelix_palette(as_cmap=True),ms = 100,dpi = dpi,  \
             figname=os.path.join(dir_figures, 'pred_actual_decompose_mean.tiff'))
    
    rmse = np.sqrt(mean_squared_error(anomaly.test_y, anomaly.pred_y))
    r2 = r2_score(anomaly.test_y, anomaly.pred_y)
    plot_pred_actual(anomaly.test_y, anomaly.pred_y, r2, rmse, \
                     xlabel = "FMC anomaly", axis_lim = [-100,100], zoom = 1,\
                     cmap = sns.cubehelix_palette(as_cmap=True),dpi = dpi, \
     figname=os.path.join(dir_figures, 'pred_actual_decompose_anomaly.tiff'))
    return anomaly
    
def groupwise_rmse(test_y, pred_y, groups):
    ind_to_forest = {0: 'crop',
           0: 'crop',
           1: 'closed needleleaf',
           2:'mixed forest',
           3:'shrub',
           3:'shrub',
           3:'shrub',
           4:'grass'}
    rmse={}
    for group in range(5):
        sub1 = test_y[groups==group]
        sub2 = pred_y[groups==group]
        rmse[group] = ind_to_forest[group]+\
           r', rmse = %02d'%np.sqrt(mean_squared_error(sub1,sub2))
    return rmse


def rename_importance_chart_labels(ax):
    labels =  [item.get_text() for item in ax.get_yticklabels(which='both')]
    new_labels = labels.copy()
    for ctr, label in enumerate(new_labels):
         if label in ['canopy_height','forest_cover']:
             continue
         elif '_smoothed' in label:
             label = label.replace('_smoothed', '')
#             new_labels[ctr] = label
             if '_' in label:
                  label = r"%s/%s"%(label.split('_')[0],label.split('_')[1])
                  new_labels[ctr] = label
         elif '_anomaly' in label: 
             label = label.replace('_anomaly', '')
#             new_labels[ctr] = label
             if '_' in label:
                  label = r"%s/%s"%(label.split('_')[0],label.split('_')[1])
             label = r'\Delta\left(%s\right)'%label
         else:
             continue
         if ('vv' in label) or ('vh' in label):
             label =label.replace('vv','\sigma_{VV}').replace('vh', '\sigma_{VH}')

         label = "$%s$"%label
         new_labels[ctr] = label
    ax.set_yticklabels(new_labels)
    return ax
        
def plot_importance(rmse_diff, model_rmse, xlabel = "RMSE Shift (%FMC anomaly)",\
                    zoom = 1, figname = None, dpi = 600):
    Df = pd.DataFrame(rmse_diff)
    Df=Df.sort_values('mean')
    Df=append_color_importance(Df)
    Df.index = Df.index.str.lower()
    Df['mean']+=model_rmse
#    Df.loc[Df.index!='elevation','mean'] = 0
#    Df['mean'] = 0
    fig, ax = plt.subplots(figsize = (zoom*5,zoom*8), dpi = dpi)
    Df['mean'].plot.barh(width=0.8,color=Df.color,xerr=Df['sd'],\
           error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1), ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Variable hidden')
    green = '#1b9e77'
    brown = '#d95f02'
    blue = '#7570b3'
    legend_elements = [Patch(facecolor=green, edgecolor=None,label='Optical'),\
                       Patch(facecolor=brown, edgecolor=None,label='Static'),\
                        Patch(facecolor=blue, edgecolor=None,label='Microwave')]
    ax.legend(handles=legend_elements,frameon=True, title='Variable Type')
#    if model_rmse
    ax.axvline(model_rmse, ls = '--',color = 'k')
    ax.annotate('$Model$  \n$RMSE$  ',\
                xy=(model_rmse,16),\
                ha='right',va='top')
    ax = rename_importance_chart_labels(ax)
    if not(figname is None):
        plt.savefig(os.path.join(dir_figures,figname), dpi = dpi,\
                    bbox_inches="tight")
    plt.show()
    return ax



def append_color_importance(Df):
    green = '#1b9e77'
    brown = '#d95f02'
    blue = '#7570b3'
    
    micro=['vv','vv_angle_corr','vh_angle_corr','vh', 'vh/ndvi','vv/ndvi','vh/vv',\
           "vh_pm","vh_am","vv_pm","vv_am",\
             "vh_pm_ndvi", "vh_am_ndvi", "vv_pm_ndvi",\
             "vv_am_ndvi", 'vv_ndvi',\
             'vh_ndvi', 'vh_red', 'vv_red','vh_blue', 'vv_blue',\
             'vh_nir', 'vv_nir',\
             'vh_green', 'vv_green']
    veg=['red','green','blue','swir','nir',
         'ndvi', 'ndwi','nirv']
    micro = micro+[x+"_smoothed" for x in micro]+[x+"_anomaly" for x in micro]
    veg = veg+[x+"_smoothed" for x in veg]+[x+"_anomaly" for x in veg]
    topo=['slope', 'elevation', 'canopy_height','latitude', "doy",
          'longitude','silt', 'sand', 'clay', 'incidence_angle',"forest_cover"]
    Df['color']=None
    Df.loc[Df.index.isin(micro),'color']=blue
    Df.loc[Df.index.isin(veg),'color']=green
    Df.loc[Df.index.isin(topo),'color']=brown
    return Df

def plot_errors_spatially(test_x, test_y, pred_y):
    ####### plot usa map
    variable = pred_y - test_y
    latlon = df.loc[test_x.index,['latitude','longitude']]
    fig, ax, m = plot_usa()
    cmap = 'RdYlGn'
    plot=m.scatter(latlon.longitude.values, latlon.latitude.values, 
                   s=20,c=variable,cmap =cmap ,edgecolor = 'k',\
                        marker='o',latlon = True, zorder = 2,\
                        vmin = -60, vmax = 60)
    plt.setp(ax.spines.values(), color='w')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(plot,ax=ax,cax=cax)
    ax.set_title('Error (% fuel moisture)')
    plt.show()

def plot_usa(enlarge = 1.): 
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(8*enlarge,5*enlarge))
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    m = Basemap(llcrnrlon=-123,llcrnrlat=23,urcrnrlon=-90,urcrnrlat=50,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-105)
    m.drawmapboundary(fill_color='lightcyan')
    #-----------------------------------------------------------------------
    # load the shapefile, use the name 'states'
    m.readshapefile('D:/Krishna/projects/vwc_from_radar/cb_2017_us_state_500k', 
                    name='states', drawbounds=True)
    statenames=[]
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        statenames.append(statename)
    for nshape,seg in enumerate(m.states):
        if statenames[nshape] == 'Alaska':
        # Alaska is too big. Scale it down to 35% first, then transate it. 
            new_seg = [(0.35*args[0] + 1100000, 0.35*args[1]-1500000) for args in seg]
            seg = new_seg    
        poly = Polygon(seg,facecolor='papayawhip',edgecolor='k', zorder  = 1)
        ax.add_patch(poly)
    return fig, ax, m

def landcover_wise_pred(test_y, pred_y):
    lc_dict = {20: 'crop',
           30: 'crop',
           70: 'closed needleleaf',
           100:'mixed forest',
           110:'shrub',
           120:'shrub',
           130:'shrub',
           140:'grass'}
    pred_y = pd.Series(pred_y, index = test_y.index)
    r2_df = pd.DataFrame([test_y, pred_y, \
                          df.loc[test_y.index,'forest_cover'].map(lc_dict)], \
        index = ['true','pred', 'forest_cover']).T
    for group in r2_df.forest_cover.unique():
        sub = r2_df.loc[r2_df.forest_cover==group,['true','pred']].astype(float)
        ax = plot_pred_actual(sub.true, sub.pred, r2_score(sub.true, sub.pred), cmap = "YlOrRd_r")
        ax.set_title(group)


def color_based_on_species(species):
    colors = species.copy()
    sagebrush=['Sagebrush, Silver','Sagebrush, Mountain Big',  'Sagebrush, Basin Big',
    'Sagebrush, California', 'Sagebrush, Black','Sagebrush, Wyoming Big', 
    'Sagebrush, Sand','Sagebrush, Bigelows', 'Sage, Black']
    pine = ['Pine, Ponderosa','Pine, Lodgepole','Pine, Interior Ponderosa',
     'Pine, Loblolly']
    chamise = ['Chamise, Old Growth','Chamise','Chamise, New Growth']
    manzanita = ['Manzanita, Whiteleaf','Manzanita, Eastwoods','Manzanita, Pinemat',
        'Manzanita, Greenleaf','Manzanita, Pointleaf']
    oak = ['Oak, Texas Red','Oak, Live','Oak, Gambel', 'Oak, Sonoran Scrub',
         'Oak, Water','Tanoak','Oak, California Live','Oak, Gray', 'Oak, Emory']
    juniper = [ 'Juniper, Redberry','Juniper, Rocky Mountain',  'Juniper, Utah',
       'Juniper, Ashe','Juniper, Creeping','Juniper, Oneseed', 'Juniper, Alligator',
       'Juniper, Western']
    ceonothus = ['Ceanothus, Whitethorn','Ceanothus, Bigpod', 'Ceanothus, Redstem',
       'Ceanothus, Desert', 'Ceanothus, Buckbrush','Ceanothus, Deerbrush',       
         'Ceanothus, Hoaryleaf',   'Ceanothus, Snowbrush']
    fir = ['Fir, California Red','Douglas-Fir', 'Fir, Subalpine',   'Fir, Grand',
              'Douglas-Fir, Coastal','Douglas-Fir, Rocky Mountain',  'Fir, White']
    others = ['Mesquite, Honey', 'Bitterbrush, Desert',
        'Red Shank','Pinyon, Twoneedle', 'Cedar, Incense', 'Pinyon, Mexican',
       'Pinyon, Singleleaf',  'Bitterbrush, Antelope',
       'Buckwheat, Eastern Mojave', 'Snowberry, Mountain',
        'Spruce, Engelmann', 'Chinquapin, Bush',
        'Tamarisk', 'Sage, Purple',
       'Coyotebrush', 'Redcedar, Eastern',  'Forage kochia',
       'Snowberry, Western', 'Fescue, Arizona',  'Maple, Rocky Mountain',
       'Yaupon', 'Duff (DC)','Bluestem, Little',
       'Pinegrass',  'Sumac, Evergreen',  'Ninebark, Pacific']
    
    colors.loc[colors.isin(sagebrush)] = 1
    colors.loc[colors.isin(pine)] = 2
    colors.loc[colors.isin(chamise)] = 3
    colors.loc[colors.isin(manzanita)] = 4
    colors.loc[colors.isin(oak)] = 5
    colors.loc[colors.isin(juniper)] = 6
    colors.loc[colors.isin(ceonothus)] = 7
    colors.loc[colors.isin(fir)] = 8
    colors = colors.convert_objects(convert_numeric=True)
    colors.loc[colors.isnull()] = 9
    return colors

def ind_to_species():
    ind_to_species = dict(zip(range(1,10), \
                ['sagebrush', 'pine','chamise','manzanita','oak','juniper','ceonothus',\
                 'fir','others']\
                             ))        
    return ind_to_species
####################################################################    ######
if __name__ == "__main__": 
    pd.set_option('display.max_columns', 30)
    sns.set(font_scale=2.1, style = 'ticks')
    
    ############################ inputs
    seed = 7
    np.random.seed(seed)
    epochs = int(1e3)
    retrain_epochs =int(1e3)
    batch_size = 2**12
    overwrite = False
    load_model = True
    save_name = 'rnn_19-mar-2019'
#    train_further = 0
    plot = 1
    response = "fm_smoothed"
    dynamic_features = ["fm_smoothed","vv_smoothed","vh_smoothed",\
                    "blue_smoothed","green_smoothed","red_smoothed","nir_smoothed",\
                    'ndvi_smoothed', 'ndwi_smoothed',\
                    'vv_ndvi_smoothed','vh_ndvi_smoothed',\
                    'vv_red_smoothed','vh_red_smoothed',\
                    'vv_nir_smoothed','vh_nir_smoothed',\
                    'vv_blue_smoothed','vh_blue_smoothed',\
                    'vv_green_smoothed','vh_green_smoothed', 'doy']
    ## only opt
    #dynamic_features = ["fm_anomaly",\
    #                    "blue_anomaly","green_anomaly","red_anomaly","nir_anomaly",\
    #                    'ndvi_anomaly', 'ndwi_anomaly',\
    #                    ]
    static_features = ['slope', 'elevation', 'canopy_height','forest_cover',
                    'silt', 'sand', 'clay', 'latitude', 'longitude']
    

    os.chdir(dir_data)
    df =make_df(dynamic_features, static_features = static_features)
    #df['vh_pm_anomaly'] = df['fm_anomaly']+10
    train_x, train_y, test_x, test_y, n_features =\
               build_data(df, dynamic_features+static_features, ignore_multi_spec = False)
    #print(len(train_y)+len(test_y))
    
    model = build_model(n_features)
    change_lr = LearningRateScheduler(scheduler)
    filepath = os.path.join(dir_codes, 'model_checkpoint/weights_%s.hdf5'%save_name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint, change_lr]
    #tbCallBack = k.callbacks.TensorBoard(log_dir='./tb_log', histogram_freq=0, write_graph=True, write_images=False)
    if  load_model&os.path.isfile(filepath):
        model.load_weights(filepath)
        # Compile model (required to make predictions)
        model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])    
        rmse_diff = pd.read_pickle(os.path.join(dir_codes, \
                    'model_checkpoint/rmse_diff_%s'%save_name))
        print('[INFO] \t Model loaded')
    else:
        if os.path.isfile(filepath):
            if not(overwrite):
                print('[INFO] File path already exists. Try Overwrite = True or change file name')
                raise
        print('[INFO] \t Retraining Model...')
        model.fit(train_x,train_y, validation_data = (test_x, test_y),  epochs=epochs, \
          batch_size=batch_size, callbacks=callbacks_list, verbose = True)
        rmse_diff, model_rmse = infer_importance(model, train_x, train_y, \
             test_x, test_y, batch_size = batch_size, retrain_epochs = retrain_epochs, \
             change_lr = change_lr, retrain = True, iterations = 10)
        rmse_diff.to_pickle(os.path.join(dir_codes, 'model_checkpoint/rmse_diff_%s'%save_name))
    pred_y = model.predict(test_x).flatten()
    model_rmse = np.sqrt(mean_squared_error(test_y, pred_y))
#    rmse_diff, model_rmse = infer_importance_by_var_category(model, train_x, train_y, \
#             test_x, test_y, batch_size = batch_size, retrain_epochs = retrain_epochs, \
#             change_lr = change_lr, retrain = False)
    ######################################################## make_plots=
    if plot:
            
        plot_pred_actual(test_y, pred_y, r2_score(test_y, pred_y), model_rmse,\
                         zoom = 1.5,dpi = 72,  \
             figname=os.path.join(dir_figures, 'pred_actual_anomaly_FMC.tiff'))
#        ax = plot_importance(rmse_diff, model_rmse, zoom = 1.5, \
#             figname=os.path.join(dir_figures, 'importance_anomaly_FMC.tiff'),\
#             xlabel = "RMSE (% FMC anomaly)", dpi = 72)


