#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:24:33 2019

@author: lisatostrams
"""

import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error

from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np


from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

chunks = pd.read_csv("XTrain.csv")
print('There are {} chunks in the file.'.format(len(chunks)))

summary = ['q01_roll_std_100', 'min_roll_std_100', 'q01_roll_std_10', 'min_roll_std_10','hmean','gmean','Hilbert_mean',
              'q01_roll_std_1000', 'min_roll_std_1000', 'max_roll_std_1000', 'mean_change_abs','iqr','iqr1','ave10',
              'mean_change_rate_first_50000', 'sum', 'q05_roll_std_10', 'q05_roll_mean_10',
              'q05_roll_std_100', 'q01_roll_mean_10', 'q05_roll_std_1000', 'mean_change_rate_last_10000',
              'mean_change_rate_last_50000', 'mean_change_rate_first_10000', 'q99', 'max_roll_std_100',
              'q75Audio', 'q25Audio', 'abs_max_roll_mean_100', 'skew', 'q01', 'abs_max_roll_mean_10',
              'abs_std', 'q95', 'q05', 'min_roll_mean_10', 'abs_trend', 'q95_roll_mean_10', 'q95_roll_std_10',
              'abs_q95', 'ave_roll_std_10', 'stdAudio', 'ave_roll_mean_10', 'ave_roll_std_100', 'ave_roll_std_1000',
              'abs_mean', 'std_roll_std_10', 'av_change_rate_roll_std_10', 'q95_roll_std_100', 'std_roll_std_100',
              'av_change_rate_roll_std_100', 'minAudio', 'max_first_10000', 'maxAudio', 'max_roll_mean_10',
              'std_roll_mean_10', 'std_first_10000', 'av_change_rate_roll_mean_10', 'mean_change_rate',
              'std_roll_std_1000', 'av_change_rate_roll_std_1000', 'q95_roll_std_1000', 'min_first_50000',
              'min_first_10000', 'min_last_10000', 'std_first_50000', 'q95_roll_mean_100', 'max_roll_mean_100',
              'q05_roll_mean_100', 'std_roll_mean_100', 'av_change_rate_roll_mean_100', 'avg_last_10000',
              'min_roll_mean_100', 'max_first_50000', 'ave_roll_mean_100', 'max_last_50000', 'std_last_50000',
              'max_last_10000', 'max_roll_std_10','std_roll_mean_1000','av_change_rate_roll_mean_1000','max_roll_mean_1000','q01_roll_mean_100',
              'min_last_50000','mad','ave_roll_mean_1000','q95_roll_mean_1000','max_to_min_diff','stdAudioIncrease', 'min_roll_mean_1000', 'q05_roll_mean_1000', 'count_big',
              'abs_max_roll_std_1000', 'abs_max_roll_std_10','abs_max_roll_std_100', 'meanAudio', 'avg_last_50000','avg_first_50000','std_last_10000','q99_roll_mean_10',
              'av_change_abs_roll_mean_1000','av_change_abs_roll_std_1000','av_change_abs_roll_mean_100','av_change_abs_roll_mean_10','av_change_abs_roll_std_100',
              'av_change_abs_roll_std_10', 'q99_roll_mean_1000','avg_first_10000','kurt','q01_roll_mean_1000','abs_q99','q99_roll_mean_100',
              'trend','q99_roll_std_10','modeAudio','q99_roll_std_1000','medianAudio','max_to_min','abs_q01','q99_roll_std_100',
              'abs_q05', 'Hann_window_mean_50', 'Hann_window_mean_150','Hann_window_mean_1500',
              'Hann_window_mean_15000','classic_sta_lta1_mean','classic_sta_lta2_mean','classic_sta_lta3_mean',
              'classic_sta_lta4_mean','classic_sta_lta5_mean','classic_sta_lta6_mean','classic_sta_lta7_mean','classic_sta_lta8_mean','autocorr_1',
              'autocorr_5','autocorr_10','autocorr_50','autocorr_100','autocorr_500','autocorr_1000','autocorr_5000','autocorr_10000','abs_max_roll_mean_1000',
              'Kalman_correction','exp_Moving_average_300_mean','exp_Moving_average_3000_mean',
              'exp_Moving_average_30000_mean','MA_700MA_std_mean','MA_700MA_BB_high_mean','MA_700MA_BB_low_mean',
              'MA_400MA_std_mean','MA_400MA_BB_high_mean','MA_400MA_BB_low_mean','MA_1000MA_std_mean','q999','q001',
              'Rmean','Rstd','Rmax','Rmin','Imean','Istd','Imax','Imin','Rmean_last_5000','Rstd__last_5000','Rmax_last_5000','Rmin_last_5000',
              'Rmean_last_15000','Rstd_last_15000','Rmax_last_15000','Rmin_last_15000']

X = chunks[summary]
y = chunks['endTime']
X=X.replace([np.inf, -np.inf], np.nan)
X=X.fillna(0)
y=y.fillna(0)
#%%
Xtrain,Xval,ytrain,yval = model_selection.train_test_split(X,y,test_size=0.5)
err = []
for d in range(1,20):
    dtc = tree.DecisionTreeRegressor(max_depth=d) #train decision tree
    dtc = dtc.fit(Xtrain,ytrain)
    y_est = dtc.predict(Xval)
    mae = np.mean(abs(y_est-yval))
    dtc = dtc.fit(Xval,yval)
    y_est = dtc.predict(Xtrain)
    mae += np.mean(abs(y_est-ytrain))
    print('dtc depth: level {} mae: {}'.format(d,mae/2))
    err.append((d,mae/2))
    
err = np.array(err)
idx = np.argmin(err[:,1])
plt.plot(err[:,0],err[:,1])
plt.title('Lowest error at level {}'.format(err[idx,0]))
plt.show()
max_depth = err[idx,0]
#%%
err = []
for d in [10,50,100,300]:
    rf = RandomForestRegressor(n_estimators = d,max_depth=5)
    rf = rf.fit(Xtrain, ytrain)
    y_est = rf.predict(Xval)
    mae = np.mean(abs(y_est-yval))
    dtc = rf.fit(Xval,yval)
    y_est = rf.predict(Xtrain)
    mae += np.mean(abs(y_est-ytrain))
    print('rf estimators: level {} mae: {}'.format(d,mae/2))
    err.append((d,mae/2))
    
err = np.array(err)
idx = np.argmin(err[:,1])
plt.plot(err[:,0],err[:,1])
plt.title('Lowest error at level {}'.format(err[idx,0]))
plt.show()
n_estimators = err[idx,0]
#%%

err = []
for d in np.linspace(0.01,10,num=200):
    reg = Ridge(alpha=d)
    reg = reg.fit(Xtrain, ytrain)

    y_est = reg.predict(Xval)
    mae = np.mean(abs(y_est-yval))
    dtc = reg.fit(Xval,yval)
    y_est = reg.predict(Xtrain)
    mae += np.mean(abs(y_est-ytrain))
    print('Linreg ridge alpha: level {} mae: {}'.format(d,mae/2))
    err.append((d,mae/2))
    
err = np.array(err)
idx = np.argmin(err[:,1])
plt.plot(err[:,0],err[:,1])
plt.title('Lowest error at level {:.6f}'.format(err[idx,0]))
plt.show()
ridge_alpha = np.round(err[idx,0],decimals=6)
#%%

err = []
for d in np.linspace(5,50,num=10):
    knn = KNeighborsRegressor(n_neighbors=int(d),algorithm='ball_tree')
    knn = knn.fit(Xtrain,ytrain)

    y_est = knn.predict(Xval)
    mae = np.mean(abs(y_est-yval))
    dtc = knn.fit(Xval,yval)
    y_est = knn.predict(Xtrain)
    mae += np.mean(abs(y_est-ytrain))
    print('KNN neighbors: level {} mae: {}'.format(d,mae/2))
    err.append((d,mae/2))
    
err = np.array(err)
idx = np.argmin(err[:,1])
plt.plot(err[:,0],err[:,1])
plt.title('Lowest error at level {}'.format(err[idx,0]))
n_neighbours = int(err[idx,0])

plt.show()

#%%

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain_norm = scaler.transform(Xtrain)
Xval_norm = scaler.transform(Xval)

err = []
for d in [0.5,0.4,0.3,0.2,1e-1,1e-2,1e-3,1e-4,1e-5]:
    svmnorm = SVR(tol=d,gamma='auto')
    svmnorm = svmnorm.fit(Xtrain_norm, ytrain)
    y_est = svmnorm.predict(Xval)
    mae = np.mean(abs(y_est-yval))
    dtc = svmnorm.fit(Xval,yval)
    y_est = svmnorm.predict(Xtrain)
    mae += np.mean(abs(y_est-ytrain))
    print('svm tol: level {} mae: {}'.format(d,mae/2))
    err.append((d,mae/2))
    
err = np.array(err)
idx = np.argmin(err[:,1])
plt.plot(err[:,0],err[:,1])
plt.title('Lowest error at level {}'.format(err[idx,0]))
tol = int(err[idx,0])
plt.show()

#%%
d_train = xgb.DMatrix(data=Xtrain_norm, label=ytrain, feature_names=Xtrain.columns)
d_val = xgb.DMatrix(data=Xval_norm, label=yval, feature_names=Xval.columns)
evallist = [(d_val, 'eval')]



err = []
for d in range(1,10):
    xgb_params = {
        'eval_metric': 'rmse',
        'seed': 1337,
        'verbosity': 0,
        'max_depth':int(d),
        'n_estimators':300,
        'silent':1,
        'gamma':1,
        'colsample_bytree':0.7,
        'min_child_weight':300
        
    }

    evallist = [(d_val, 'eval')]

    model = xgb.train(dtrain=d_train, num_boost_round=2000, evals=evallist, early_stopping_rounds=50,  params=xgb_params)

    y_est = model.predict(d_val, ntree_limit=model.best_ntree_limit)
    mae = np.mean(abs(y_est-yval))
    evallist = [(d_train, 'eval')]

    model = xgb.train(dtrain=d_val, num_boost_round=2000, evals=evallist, early_stopping_rounds=50,  params=xgb_params)
    y_est = model.predict(d_train, ntree_limit=model.best_ntree_limit)
    mae += np.mean(abs(y_est-ytrain))
    print('Xgb depth: level {} mae: {}'.format(d,mae/2))
    err.append((d,mae/2))
    
err = np.array(err)
idx = np.argmin(err[:,1])
plt.plot(err[:,0],err[:,1])
plt.title('Lowest error at level {}'.format(err[idx,0]))
xgb_depth = int(err[idx,0])

plt.show()


#%%

#
#from catboost import CatBoostRegressor
#
#
#err = []
#for d in np.linspace(0,1,11):
#    Cat = CatBoostRegressor(iterations=400,
#                               depth=d,
#                               learning_rate=0.1,
#                               loss_function= 'RMSE',
#                               verbose=False
#                               )
#
#    Cat.fit(Xtrain, ytrain)
#    y_est = Cat.predict(Xval)
#    mae = np.mean(abs(y_est-yval))
#    dtc = Cat.fit(Xval,yval)
#    y_est = Cat.predict(Xtrain)
#    mae += np.mean(abs(y_est-ytrain))
#    print('Cat depth: level {} mae: {}'.format(d,mae/2))
#    err.append((d,mae/2))
#    
#err = np.array(err)
#idx = np.argmin(err[:,1])
#plt.plot(err[:,0],err[:,1])
#plt.title('Lowest error at level {}'.format(err[idx,0]))
#cb_alpha = int(err[idx,0])
#plt.show()