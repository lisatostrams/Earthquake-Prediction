# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:14:55 2019

@author: pepij
"""


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv1D
import pandas as pd
from sklearn import tree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
from tpot import TPOTRegressor
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb

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

Xtrain,Xval,ytrain,yval = model_selection.train_test_split(X,y,test_size=0.25)

Test = pd.read_csv('Xtest.csv')
Xtest = Test[summary]
#%%
model_m = Sequential()
model_m.add(Conv1D(100, 10, activation='relu', input_shape=())
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())
#%%