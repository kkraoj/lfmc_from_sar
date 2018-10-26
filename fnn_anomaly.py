# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 18:59:11 2018

@author: kkrao
"""
## computation
import os
import numpy as np
import pandas as pd
##nn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import keras as k
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras import regularizers
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


pd.set_option('display.max_columns', 30)
sns.set(font_scale=1.5, style = 'ticks')

############################ inputs
seed = 7
np.random.seed(seed)
epochs = int(1e3)
batch_size = 2**12
pct_thresh = 20
response = "fm_anomaly"
dynamic_features = ["fm_anomaly","vh_pm_anomaly","vh_am_anomaly",\
                    "blue_anomaly","green_anomaly","red_anomaly","nir_anomaly",\
                    'ndvi_anomaly', 'ndwi_anomaly']

static_features = ['slope', 'elevation', 'canopy_height','forest_cover',
                'silt', 'sand', 'clay', 'latitude', 'longitude']
#########################
os.chdir('D:/Krishna/projects/vwc_from_radar')
#df = pd.read_pickle('data/df_all')

def make_df(features):
    pure_species_sites = ['Clark Motorway, Malibu',
                         'Corralitos New Growth',
                         'Corralitos Old Growth',
                         'Glendora Ridge, Glendora',
                         'Mt. Woodson Station',
                         'Pebble Beach New Growth',
                         'Pebble Beach Old Growth',
                         'Ponderosa Basin Ceanothus, Buckbrush New',
                         'Ponderosa Basin Ceanothus, Buckbrush Old',
                         'Ponderosa Basin Manzanita, Whiteleaf New',
                         'Ponderosa Basin Manzanita, Whiteleaf Old',
                         'ST RTE 35 & 92',
                         'Tapo Canyon, Simi Valley',
                         'Throckmorton',
                         'Trippet Ranch, Topanga']
    pure_species_sites = ['Trippet Ranch, Topanga']
    Df = pd.DataFrame()
    for feature in features:
        df = pd.read_pickle('data/%s'%feature).loc[:,pure_species_sites].stack()
        df.name = feature
        Df = pd.concat([Df,df],axis = 'columns')
    Df.dropna(inplace = True)
    Df.head()
    Df.index.names= ['date','site']
    Df.reset_index(inplace=True)
    static_features = pd.read_csv('data/static_features.csv',dtype = {'site':str}, index_col = 0)
    Df.site = Df.site.values
    Df = Df.join(static_features, on = 'site')
    Df.reset_index(inplace=True, drop = True)
    Df.drop(['site','date'],axis = 1, inplace = True)
    Df = Df.astype('float16')
    return Df
    

def build_data(df, columns, response = response,ignore_multi_spec = False, \
               pct_thresh = pct_thresh):
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
    df.dropna(inplace = True)
    dont_norm = df[response]
#    df = (df - df.mean())/(df.std())    #normalize 
    df[response] = dont_norm
    train, test = train_test_split(df, train_size = 0.6, test_size = 0.4)
    train_y= train[response]
    train_x= train.drop([response],axis = 'columns')
    test_y= test[response]
    test_x= test.drop([response],axis = 'columns')
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
    if epoch%50 == 0:
        old_lr = K.get_value(model.optimizer.lr)
        new_lr = old_lr*lr_decay
        if new_lr<=1e-4:
            return  K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, new_lr)
    return K.get_value(model.optimizer.lr)

def build_model(n_features):
	# create model
    reg = regularizers.l1(1e-2)
    model = Sequential()
    model.add(Dense(15, input_dim = n_features,kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(150, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(15, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(60, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
#    model.add(Dense(30, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
    model.add(Dense(30, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
    model.add(Dense(15, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
    model.add(Dense(30, kernel_initializer='normal', activation='relu',kernel_regularizer=reg))
    model.add(Dense(1, kernel_initializer='normal',kernel_regularizer=reg))
#    sgd = Adam(lr=1e-1)
#    sgd = SGD(lr=1e-2, momentum=0.9, decay=1e-4, nesterov=True)
    sgd = Adam(lr=3e-2, beta_1=0.9, beta_2=0.99, epsilon=None, decay=1e-5, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

def infer_importance(model, train_x, train_y, test_x, test_y, \
                     retrain = False, iterations =10):
    pred_y = model.predict(test_x).flatten()
    rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    rmse_diff = pd.DataFrame(index = test_x.columns,\
                             columns = range(iterations))
    for itr in range(iterations):
        for feature in test_x.columns:
            if retrain:
                sample_train_x = train_x.copy()
                sample_train_x.loc[:,feature] = 0.
                model.fit(sample_train_x,train_y, epochs=epochs, batch_size=batch_size, \
                          callbacks=[change_lr], verbose = False)
            sample_test_x = test_x.copy()
            sample_test_x.loc[:,feature] = 0.
            sample_pred_y = model.predict(sample_test_x).flatten()
            sample_rmse = np.sqrt(mean_squared_error(test_y, sample_pred_y))
            rmse_diff.loc[feature, itr] = sample_rmse - rmse
    rmse_diff['mean'] = rmse_diff.mean(axis = 'columns')
    rmse_diff['sd'] = rmse_diff.drop('mean',axis = 'columns').std(axis = 'columns')
    rmse_diff.drop(range(iterations),axis = 'columns', inplace = True)
#    print(rmse_diff)
    return rmse_diff
        
####################################################################    ######
df =make_df(dynamic_features)
#df['vh_pm_anomaly'] = df['fm_anomaly']+10
train_x, train_y, test_x, test_y, n_features =\
           build_data(df, dynamic_features+static_features, ignore_multi_spec = False)
#print(len(train_y)+len(test_y))
model = build_model(n_features)
change_lr = k.callbacks.LearningRateScheduler(scheduler)
#tbCallBack = k.callbacks.TensorBoard(log_dir='./tb_log', histogram_freq=0, write_graph=True, write_images=False)
model.fit(train_x,train_y, epochs=epochs, batch_size=batch_size, callbacks=[change_lr])
pred_y = model.predict(test_x).flatten()
print('Train score: %.2f' % r2_score(train_y,model.predict(train_x) ))
print('Test score: %.2f' % r2_score(test_y, pred_y))
rmse_diff = infer_importance(model, train_x, train_y, test_x, test_y, retrain = True)

######################################################## make_plots
def plot_pred_actual(test_y, pred_y, R2):
    fig, ax = plt.subplots(figsize = (3.5,3.5), dpi = 200)
    plt.axis('scaled')
    x = test_y
    y = pred_y
    #x=train_y
    #y = model.predict(train_x).flatten()
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
#    groups = color_based_on_species(df.loc[test_x.index,'Fuel'])
    plot = ax.scatter(x,y, c=z, s=8, edgecolor='', cmap = "viridis")
    axis_lim = [-25,50]
    ax.set_xlim(axis_lim)
    ax.set_ylim(axis_lim)
    ax.plot(axis_lim,axis_lim, lw =.5, color = 'grey')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel('Predicted FM anomaly')
    ax.set_xlabel('Actual FM anomaly')
#    ax.set_xticks([-50,0,50,100])
#    ax.set_yticks([-50,0,50,100])
    ax.annotate('$R^2_{test}=%0.2f$'%R2, xy=(0.15, 0.95), xycoords='axes fraction',\
                ha='left',va='top')
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.08)
#    fig.colorbar(plot,ax=ax,cax=cax)
#    ticklabels = groupwise_rmse(test_y, pred_y, groups)
#    cax.set_yticklabels(ticklabels.values())
#    cax.set_ylabel('Canopy height (m)')
    plt.show()
    
def groupwise_rmse(test_y, pred_y, groups):
    rmse={}
    for group in range(1,10):
        sub1 = test_y[groups==group]
        sub2 = pred_y[groups==group]
        rmse[group] = ind_to_species[group]+\
           r', rmse = %02d'%np.sqrt(mean_squared_error(sub1,sub2))
    return rmse
        
    
def plot_importance(rmse_diff):
    Df = pd.DataFrame(rmse_diff)
    Df=Df.sort_values('mean')
    Df=append_color_importance(Df)
    Df.index = Df.index.str.lower()

    fig, ax = plt.subplots(figsize = (3,5))
    Df['mean'].plot.barh(width=0.8,color=Df.color,xerr=Df['sd'],\
           error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1), ax=ax)
    ax.set_xlabel('RMSE Shift (% FM anomaly)')
    green = '#1b9e77'
    brown = '#d95f02'
    blue = '#7570b3'
    legend_elements = [Patch(facecolor=green, edgecolor=None,label='Optical'),\
                       Patch(facecolor=brown, edgecolor=None,label='Structure'),\
                        Patch(facecolor=blue, edgecolor=None,label='Microwave')]
    ax.legend(handles=legend_elements,frameon=True, title='Variable Type')
#    plt.tight_layout()
#    print(Df)
    return Df

def append_color_importance(Df):
    green = '#1b9e77'
    brown = '#d95f02'
    blue = '#7570b3'
    micro=['vv','vv_angle_corr','vh_angle_corr','vh', 'vh/ndvi','vv/ndvi','vh/vv',\
           "vh_pm_anomaly","vh_am_anomaly","vv_pm_anomaly","vv_am_anomaly",\
                    ]
    veg=['red','green','blue','swir','nir','ndvi_anomaly', 'ndwi_anomaly',\
         'ndvi', 'ndwi','nirv',"blue_anomaly","green_anomaly","red_anomaly",\
         "nir_anomaly"]
    topo=['slope', 'elevation', 'canopy_height','latitude', 
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

ind_to_species = dict(zip(range(1,10), \
            ['sagebrush', 'pine','chamise','manzanita','oak','juniper','ceonothus',\
             'fir','others']\
                         ))
#####################################fun zone##################
plot_pred_actual(test_y, pred_y, r2_score(test_y, pred_y))
plot_importance(rmse_diff)
#plot_errors_spatially(test_x, test_y, pred_y)
