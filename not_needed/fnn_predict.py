# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:15:16 2019

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
from scipy import optimize
from scipy.optimize import least_squares
### plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import seaborn as sns
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap



pd.set_option('display.max_columns', 30)
sns.set(font_scale=2.1, style = 'ticks')
    

def make_df(dynamic_features, static_features,\
            data_loc = 'timeseries/',sites=None):

    Df = pd.DataFrame()
    
    for feature in dynamic_features:
        df = pd.read_pickle('%s%s'%(data_loc,feature))
        if sites is None:
            sites = df.columns
        df = df.loc[:,sites].stack(dropna=False)
        df.name = feature
        Df = pd.concat([Df,df],axis = 'columns')
    Df.fillna(0, inplace = True)
    
    Df.index.names= ['date','site']
    Df.reset_index(inplace=True)
    Df.site = Df.site.values
    static_features_all = pd.read_csv('static_features.csv',dtype = {'site':str}, index_col = 0)
    if not(static_features is None):
        static_features_subset = static_features_all.loc[:,static_features]
        Df = Df.join(static_features_subset, on = 'site')
    Df.reset_index(inplace=True, drop = True)
    Df.date = pd.to_datetime(Df.date)
#    Df.drop(['site','date'],axis = 1, inplace = True)
    Df = Df.infer_objects()
#    Df.apply(pd.to_numeric, errors='ignore')
    #### ignore single species sites
    return Df
        

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


def plot_pred_actual(test_y, pred_y, R2, model_rmse,  cmap = 'plasma', axis_lim = [-25,50],\
                      zoom = 1,\
                     figname = None, dpi = 600,ms = 8,\
                     xlabel = None, ylabel= None):
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
    
    
    ax.plot(axis_lim,axis_lim, lw =.5, color = 'grey')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
#    ax.set_ylabel('$\sum w_i\ x\ FMC_i$')
#    ax.set_xlabel('$\hat{FMC}$')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(axis_lim)
    ax.set_ylim(axis_lim)
    ax.set_yticks(ax.get_xticks())
#    ax.set_xticks([-50,0,50,100])
#    ax.set_yticks([-50,0,50,100])
    ax.annotate('$R^2=%0.2f$\n$RMSE=%0.1f$'%(np.floor(R2*100)/100, model_rmse), \
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

def multiplying_function(weights,percent, pred):
    return weights*percent-pred
    
    

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
    target = true_sub.groupby(true_sub.index).pred.mean()
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


####################################################################    ######
#if __name__ == "__main__": 
    ############################ inputs
seed = 7
np.random.seed(seed)
retrain_epochs =int(1e3)
save_name = 'smoothed_all_sites_11_mar_2019_with_doy'
#    train_further = 0
plot = False
dynamic_features = ["vv_smoothed","vh_smoothed",\
                "blue_smoothed","green_smoothed","red_smoothed","nir_smoothed",\
                'ndvi_smoothed', 'ndwi_smoothed',\
                'vv_ndvi_smoothed','vh_ndvi_smoothed',\
                'vv_red_smoothed','vh_red_smoothed',\
                'vv_nir_smoothed','vh_nir_smoothed',\
                'vv_blue_smoothed','vh_blue_smoothed',\
                'vv_green_smoothed','vh_green_smoothed', 'doy']
static_features = ['slope', 'elevation', 'canopy_height','forest_cover',
                'silt', 'sand', 'clay', 'latitude', 'longitude']
os.chdir(dir_data)
df =make_df(dynamic_features, static_features = static_features)

model = build_model(n_features = len(dynamic_features)+len(static_features))
filepath = os.path.join(dir_codes, 'model_checkpoint/weights_%s.hdf5'%save_name)
model.load_weights(filepath)
# Compile model (required to make predictions)
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])    
#print('[INFO] \t Model loaded')
df['pred'] = model.predict(df.drop(['site','date'],axis = 1)).flatten()
df.date = pd.to_datetime(df.date)
pred_sites = df.site.unique()


###########################################################################
true = pd.read_pickle("df_vwc_historic")
true.date = pd.to_datetime(true.date)
true = true.loc[true.meas_date.dt.year>=2015]
true = true.loc[~true.fuel.isin(['1-Hour','10-Hour','100-Hour', '1000-Hour'])] 
true = true[~true.fuel.str.contains("Duff")] ###dropping all dead DMC
true = true[~true.fuel.str.contains("Dead")] ###dropping all dead DMC
true = true[~true.fuel.str.contains("DMC")] ###dropping all dead DMC
no_of_species_in_sites = true.groupby('site').fuel.unique().apply(lambda series: len(series))
sites_with_leq_4_species = no_of_species_in_sites.loc[(no_of_species_in_sites<=4)].index
true = true.loc[true.site.isin(sites_with_leq_4_species)]
true.index = pd.to_datetime(true.meas_date)
ctr_found = 0
ctr_notfound = 0
ctr_more_than_1_fuels = 0


W = pd.DataFrame(index = true.site.unique(), columns = ['W1','W2','W3','W4','RMSE'])
W = W.sort_index()
col_list = ['site','date','S1','S2','S3','S4','W1','W2','W3','W4','FMC_weighted','FMC_mean','FMC_hat','RMSE']
optimized_df = pd.DataFrame(columns = col_list)
for site in true.site.unique():
    if site in pred_sites:
#            print('[INFO] Site found')
        ctr_found+=1
        true_sub = true.loc[true.site==site,['site','date','fuel','percent']]
        pred_sub = df.loc[df.site==site,['date','pred']].copy()
        true_sub = true_sub.set_index('date').join(pred_sub.set_index('date'))
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
ax = plot_pred_actual(optimized_df['FMC_hat'], optimized_df['FMC_weighted'], \
                 r2_score(optimized_df['FMC_hat'], optimized_df['FMC_weighted']),\
                mean_squared_error(optimized_df['FMC_weighted'], optimized_df['FMC_hat'])**0.5,\
                xlabel = '$\hat{FMC}$', ylabel ='$\sum w_i\ x\ FMC_i$' ,\
                     zoom = 1.5,dpi = 150, axis_lim = [0,300])    

############## pred FMC vs mean FMC (simple averaged)
ax = plot_pred_actual(optimized_df['FMC_mean'], optimized_df['FMC_hat'], \
                 r2_score(optimized_df['FMC_mean'], optimized_df['FMC_hat']),\
                 mean_squared_error(optimized_df['FMC_mean'], optimized_df['FMC_hat'])**0.5,\
                     zoom = 1.5,dpi = 150, axis_lim = [0,300], \
                     xlabel = '$\overline{FMC}$', ylabel = '$\hat{FMC}$')

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