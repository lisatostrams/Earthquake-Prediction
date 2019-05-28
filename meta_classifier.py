# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:05:09 2019

@author: Lisa
"""
import pandas as pd
from sklearn import tree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
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
from tpot import TPOTRegressor
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from catboost import CatBoostRegressor, Pool
from sklearn.feature_selection import SelectPercentile, f_regression

#use certain attributes only for dtc, rf, ridge,knn, svr and general:

dtc_attributes = ['q05_roll_std_10', 'q05_roll_std_100']
rf_attributes = ['q05_roll_std_100','q05_roll_std_10', 'q01_roll_std_100', 'q01_roll_std_10', 'q05_roll_std_1000',
                 'autocorr_5', 'q95_roll_std_10', 'mean_change_rate', 'std_roll_mean_1000', 'autocorr_50', 'avg_last_50000',
                 'std_roll_mean_100', 'exp_Moving_average_30000_mean', 'Rstd', 'q99_roll_mean_10', 'av_change_abs_roll_mean_1000',
                 'classic_sta_lta8_mean', 'avg_first_50000', 'q01_roll_mean_1000', 'autocorr_5000', 'avg_first_10000',
                 'av_change_abs_roll_std_10', 'mean_change_rate_first_50000', 'q01_roll_mean_10', 'classic_sta_lta6_mean',
                 'classic_sta_lta1_mean', 'max_roll_std_100', 'avg_last_10000', 'skew', 'q99_roll_mean_1000', 'q95',
                 'q99_roll_mean_100', 'min_roll_mean_100', 'MA_400MA_BB_high_mean', 'max_to_min_diff', 'stdAudioIncrease']
ridge_attributes = ['abs_mean', 'q05_roll_std_100', 'std_roll_std_100', 'abs_std', 'Hilbert_mean', 'gmean', 'q01_roll_std_100', 'autocorr_10', 'mean_change_rate_first_10000', 'mean_change_rate', 'q05_roll_std_10', 'hmean', 'mean_change_rate_first_50000', 'q01_roll_std_10', 'autocorr_50', 'mean_change_rate_last_50000', 'autocorr_100', 'q05_roll_mean_100', 'classic_sta_lta5_mean', 'q95_roll_std_10', 'stdAudioIncrease', 'std_roll_mean_100', 'q95_roll_mean_100', 'q05_roll_mean_10', 'autocorr_5', 'mad', 'q95_roll_mean_1000', 'autocorr_1', 'mean_change_rate_last_10000', 'q01_roll_mean_1000', 'q99_roll_mean_100', 'min_roll_std_100', 'MA_400MA_BB_low_mean', 'MA_400MA_BB_high_mean', 'avg_first_50000', 'q95_roll_std_100', 'std_roll_std_10']
knn_attributes = ['abs_mean', 'q05_roll_std_100']
svr_attributes = ['abs_mean', 'q05_roll_std_100', 'std_roll_std_100', 'abs_std', 'Hilbert_mean', 'gmean', 'q01_roll_std_100', 'autocorr_10', 'mean_change_rate_first_10000', 'mean_change_rate', 'q05_roll_std_10', 'hmean', 'mean_change_rate_first_50000', 'q01_roll_std_10', 'autocorr_50', 'mean_change_rate_last_50000', 'autocorr_100', 'q05_roll_mean_100', 'classic_sta_lta5_mean', 'q95_roll_std_10', 'stdAudioIncrease', 'std_roll_mean_100', 'q95_roll_mean_100', 'q05_roll_mean_10', 'autocorr_5', 'mad', 'q95_roll_mean_1000', 'autocorr_1', 'mean_change_rate_last_10000', 'q01_roll_mean_1000', 'q99_roll_mean_100', 'min_roll_std_100', 'MA_400MA_BB_low_mean', 'MA_400MA_BB_high_mean', 'avg_first_50000', 'q95_roll_std_100', 'std_roll_std_10', 'q95_roll_mean_10', 'classic_sta_lta1_mean', 'avg_last_50000', 'ave_roll_std_100', 'q05_roll_std_1000', 'q01_roll_mean_100', 'classic_sta_lta4_mean', 'q99_roll_mean_1000', 'autocorr_10000', 'MA_1000MA_std_mean', 'ave_roll_std_1000', 'exp_Moving_average_30000_mean', 'std_roll_mean_10', 'classic_sta_lta6_mean', 'abs_q05', 'ave_roll_std_10', 'max_to_min', 'q05_roll_mean_1000', 'medianAudio', 'q25Audio', 'MA_700MA_BB_high_mean', 'classic_sta_lta3_mean', 'q99_roll_mean_10', 'MA_700MA_BB_low_mean', 'q75Audio', 'min_roll_std_10', 'MA_400MA_std_mean', 'skew', 'ave10', 'classic_sta_lta7_mean', 'q01_roll_mean_10', 'autocorr_500', 'autocorr_1000', 'min_roll_std_1000', 'autocorr_5000', 'classic_sta_lta2_mean', 'Kalman_correction', 'q01_roll_std_1000', 'avg_first_10000', 'modeAudio', 'MA_700MA_std_mean', 'iqr1', 'std_roll_std_1000', 'classic_sta_lta8_mean', 'stdAudio', 'Hann_window_mean_15000', 'q95', 'q001', 'max_roll_mean_100', 'q99_roll_std_10', 'avg_last_10000', 'q95_roll_std_1000', 'abs_q95', 'std_last_10000', 'q99_roll_std_100', 'q99', 'min_roll_mean_100', 'std_first_10000', 'std_last_50000']
general_attributes = ['av_change_abs_roll_std_100', 'classic_sta_lta1_mean', 'av_change_abs_roll_std_1000', 'classic_sta_lta7_mean', 'abs_trend', 'classic_sta_lta4_mean', 'std_last_10000', 'mean_change_rate_first_10000', 'kurt', 'classic_sta_lta3_mean', 'max_last_50000', 'classic_sta_lta8_mean', 'autocorr_5', 'mean_change_rate', 'min_roll_mean_100', 'Rmin_last_5000', 'min_roll_std_1000', 'stdAudioIncrease', 'min_last_10000', 'max_roll_std_100', 'autocorr_50', 'Rmax_last_5000', 'mean_change_rate_first_50000', 'autocorr_500', 'autocorr_10000', 'classic_sta_lta2_mean', 'std_roll_mean_100', 'max_roll_std_1000', 'av_change_abs_roll_mean_10', 'min_roll_std_100', 'q99_roll_mean_1000', 'Imean', 'av_change_rate_roll_mean_10', 'std_roll_mean_1000', 'autocorr_100', 'av_change_rate_roll_std_1000', 'autocorr_10', 'Rstd__last_5000', 'avg_last_10000', 'autocorr_1000', 'abs_max_roll_std_1000', 'hmean', 'q01_roll_std_1000', 'abs_max_roll_mean_10', 'mean_change_rate_last_50000', 'autocorr_5000', 'abs_max_roll_mean_1000', 'ave_roll_mean_100', 'max_roll_mean_100', 'avg_first_50000', 'mean_change_rate_last_10000', 'classic_sta_lta6_mean', 'q99_roll_std_1000', 'min_last_50000', 'Rmean', 'min_roll_std_10', 'q95_roll_std_1000', 'iqr', 'abs_std', 'max_roll_mean_10', 'Rmean_last_15000', 'q01_roll_mean_1000', 'mean_change_abs', 'abs_max_roll_std_100', 'max_first_10000', 'MA_700MA_BB_high_mean', 'q05_roll_std_100', 'max_first_50000', 'av_change_rate_roll_mean_100', 'std_roll_std_10', 'exp_Moving_average_3000_mean', 'abs_max_roll_std_10', 'Hann_window_mean_50', 'q95_roll_mean_10', 'Rmin_last_15000', 'av_change_abs_roll_std_10', 'min_roll_mean_1000', 'max_to_min', 'q99_roll_mean_100', 'min_roll_mean_10', 'trend', 'q95_roll_std_100', 'avg_last_50000', 'q01_roll_std_100', 'std_first_10000', 'av_change_abs_roll_mean_1000', 'skew', 'q99', 'Hilbert_mean', 'Rmean_last_5000', 'minAudio', 'MA_700MA_BB_low_mean', 'q95_roll_mean_1000', 'av_change_rate_roll_std_10', 'q01_roll_mean_100', 'av_change_abs_roll_mean_100', 'q99_roll_std_10', 'classic_sta_lta5_mean', 'q99_roll_mean_10', 'q05_roll_mean_1000', 'Hann_window_mean_150', 'ave_roll_mean_10', 'abs_q99', 'av_change_rate_roll_std_100', 'min_first_10000', 'exp_Moving_average_300_mean', 'q001', 'Kalman_correction', 'std_last_50000', 'abs_mean', 'q01_roll_std_10', 'gmean', 'iqr1', 'ave10', 'sum', 'q05_roll_std_10', 'q05_roll_mean_10', 'q01_roll_mean_10', 'q05_roll_std_1000', 'q75Audio', 'q25Audio', 'abs_max_roll_mean_100', 'q01', 'q95', 'q05', 'q95_roll_std_10', 'abs_q95', 'ave_roll_std_10', 'stdAudio', 'ave_roll_std_100']

#%%
# define the chunks and the features:
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

Xtrain,Xval,ytrain,yval = model_selection.train_test_split(X,y,test_size=0.4)

Test = pd.read_csv('Xtest.csv')
Xtest = Test[summary]

scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain_norm = scaler.transform(Xtrain)
Xval_norm = scaler.transform(Xval)
Xtest_norm = scaler.transform(Xtest)
ytrain_est = []
yval_est = []
predictions = []

classifiers = 'DTC RF REG KNN SVM SVMlinear BRR HR XGB CAT ADA LGBM TPOT'.split(sep=' ')
#%% train all classifiers
max_depth = 3
n_estimators = 50
tol = 0.1
        
predictions = np.zeros((len(Xtest),len(classifiers)))
ytrain_est = np.zeros((len(Xtrain),len(classifiers)))
yval_est = np.zeros((len(Xval),len(classifiers)))


dtc = tree.DecisionTreeRegressor(max_depth=max_depth) #train decision tree
dtc = dtc.fit(Xtrain[dtc_attributes],ytrain)
predictions[:,0] = dtc.predict(Xtest[dtc_attributes])
ytrain_est[:,0] = dtc.predict(Xtrain[dtc_attributes])
yval_est[:,0] = dtc.predict(Xval[dtc_attributes])

rf = RandomForestRegressor(n_estimators = n_estimators)
rf = rf.fit(Xtrain[rf_attributes], ytrain)
predictions[:,1] = rf.predict(Xtest[rf_attributes])
ytrain_est[:,1] = rf.predict(Xtrain[rf_attributes])
yval_est[:,1] = rf.predict(Xval[rf_attributes])

reg = Ridge(alpha = 2)
reg = reg.fit(Xtrain[ridge_attributes], ytrain)
predictions[:,2] = reg.predict(Xtest[ridge_attributes])
ytrain_est[:,2] = reg.predict(Xtrain[ridge_attributes])
yval_est[:,2] = reg.predict(Xval[ridge_attributes])

knn = KNeighborsRegressor(n_neighbors=45,algorithm='ball_tree')
knn = knn.fit(Xtrain[knn_attributes],ytrain)
predictions[:,3] = knn.predict(Xtest[knn_attributes])
ytrain_est[:,3] = knn.predict(Xtrain[knn_attributes])
yval_est[:,3] = knn.predict(Xval[knn_attributes])

scaler = preprocessing.StandardScaler().fit(Xtrain[svr_attributes])
Xtrain_norm = scaler.transform(Xtrain[svr_attributes])
Xval_norm = scaler.transform(Xval[svr_attributes])
Xtest_norm = scaler.transform(Xtest[svr_attributes])

svmnorm = SVR(tol=tol,gamma='auto')
svmnorm = svmnorm.fit(Xtrain_norm, ytrain)
predictions[:,4] = svmnorm.predict(Xtest_norm)
ytrain_est[:,4] = svmnorm.predict(Xtrain_norm)
yval_est[:,4] = svmnorm.predict(Xval_norm)

svmlnorm = LinearSVR(max_iter=10000)
svmlnorm = svmlnorm.fit(Xtrain_norm,ytrain)
predictions[:,5] = svmlnorm.predict(Xtest_norm)
ytrain_est[:,5] = svmlnorm.predict(Xtrain_norm)
yval_est[:,5] = svmlnorm.predict(Xval_norm)

print("processing classifiers, half way")


gnb = BayesianRidge()
gnb = gnb.fit(Xtrain_norm, ytrain)
predictions[:,6] = gnb.predict(Xtest_norm)
ytrain_est[:,6] = gnb.predict(Xtrain_norm)
yval_est[:,6] = gnb.predict(Xval_norm)

hr = HuberRegressor()
hr = hr.fit(Xtrain_norm, ytrain)
predictions[:,7] = hr.predict(Xtest_norm)
ytrain_est[:,7] = hr.predict(Xtrain_norm)
yval_est[:,7] = hr.predict(Xval_norm)

xgb_params = {
        'eta': 0.03,
        'max_depth': 8,
        'subsample': 0.75,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': True,
        'nthread': 4
}
scaler = preprocessing.StandardScaler().fit(Xtrain[general_attributes])
Xtrain_norm = scaler.transform(Xtrain[general_attributes])
Xval_norm = scaler.transform(Xval[general_attributes])
Xtest_norm = scaler.transform(Xtest[general_attributes])
print("xgb fit")
d_train = xgb.DMatrix(data=Xtrain_norm, label=ytrain, feature_names=Xtrain[general_attributes].columns)
d_val = xgb.DMatrix(data=Xval_norm, label=yval, feature_names=Xval[general_attributes].columns)
evallist = [(d_val, 'eval'), (d_train, 'train')]
model = xgb.train(dtrain=d_train, num_boost_round=120, evals=evallist, early_stopping_rounds=10,  params=xgb_params)
predictions[:,8] = model.predict(xgb.DMatrix(data=Xtest_norm, feature_names=Xtest[general_attributes].columns), ntree_limit=model.best_ntree_limit)
ytrain_est[:,8] = model.predict(d_train, ntree_limit=model.best_ntree_limit)
yval_est[:,8] = model.predict(d_val, ntree_limit=model.best_ntree_limit)

print("ada fit")
abc = AdaBoostRegressor(n_estimators = 16, learning_rate = 0.05)
abc = abc.fit(Xtrain ,ytrain)
predictions[:,9] = abc.predict(Xtest)
ytrain_est[:,9] = abc.predict(Xtrain)
yval_est[:,9] = abc.predict(Xval)
print("val score: ", np.mean(abs(abc.predict(Xval)-yval)))
print("train_score: ", np.mean((abs(abc.predict(Xtrain)-ytrain))))


print("cat fit")
Cat = CatBoostRegressor(n_estimators=25000, verbose=-1, objective="MAE", loss_function="MAE", boosting_type="Ordered")
Cat.fit(Xtrain, 
              ytrain, 
              eval_set=[(Xval, yval)], 
#               eval_metric='mae',
              verbose=2500, 
              early_stopping_rounds=500)
predictions[:,10] = Cat.predict(Xtest)
ytrain_est[:,10] = Cat.predict(Xtrain)
yval_est[:,10] = Cat.predict(Xval)
print("gbm fit")
lgb_train = lgb.Dataset(Xtrain, ytrain)
lgb_eval = lgb.Dataset(Xval, yval, reference=lgb_train)
params = {'num_leaves': 51,
         'min_data_in_leaf': 10, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": -1,
         "random_state": 42}
print("train lgb")
gbm = lgb.LGBMRegressor(**params, n_estimators = 20000, n_jobs = -1)
gbm.fit(Xtrain, 
          ytrain, 
          eval_set=[(Xtrain, ytrain), (Xval, yval)], 
          eval_metric='mae',
          verbose=1000, 
          early_stopping_rounds=500)

predictions[:,11] = gbm.predict(Xtest)
ytrain_est[:,11] = gbm.predict(Xtrain)
yval_est[:,11] = gbm.predict(Xval)
print("val score: ", np.mean(abs(gbm.predict(Xval)-yval)))
print("train_score: ", np.mean((abs(gbm.predict(Xtrain)-ytrain))))
#kernel = np.var(y)* RBF(length_scale=1)
#gp = GaussianProcessRegressor(kernel=kernel,alpha=0.1).fit(Xtrain_norm, ytrain)
#predictions[:,-1] = gp.predict(Xtest_norm)
#ytrain_est[:,-1] = gp.predict(Xtrain_norm)
#yval_est[:,-1] = gp.predict(Xval_norm)


Tp = make_pipeline(
StandardScaler(),
ExtraTreesRegressor(bootstrap=True, max_features=1.0, min_samples_leaf=9, min_samples_split=7, n_estimators=100)
)
Tp.fit(Xtrain, ytrain)
print("val score: ", np.mean(abs(Tp.predict(Xval)-yval)))
print("train_score: ", np.mean((abs(Tp.predict(Xtrain)-ytrain))))

predictions[:,12] = Tp.predict(Xtest)
ytrain_est[:,12] = Tp.predict(Xtrain)
yval_est[:,12] = Tp.predict(Xval)


#%%
# Use the TPOT regressor
import datetime    

def test_tpot_performance(test_rounds):  
    scores = []
    current_score = 100
    train_scores = []
    classifiers.append('Tpot')
    old_max = 100
    for i in range(20, 100, 20):  
        print("round: ", i,"/",test_rounds)
        print("start_time = " , datetime.datetime.now().time())
        Tp = TPOTRegressor(max_time_mins =i)
        Tp.fit(Xtrain, ytrain)
        current_score = np.mean(abs(Tp.predict(Xval)-yval))
        train_scores.append(np.mean((abs(Tp.predict(Xtrain)-ytrain))))
        print("score = ", current_score)
        print("current: ", scores)
        if current_score < old_max:
            print("better tpot score")
            Tp.export('tpot_exported_pipeline.py')
            old_max = current_score
        scores.append(current_score)
    plt.subplot(121)
    plt.title('valuation scores')
    plt.plot(scores)
    plt.savefig('Plots/eval_scores.png')
    plt.subplot(122)
    plt.title('Plots/training scores')
    plt.plot(train_scores)
    plt.savefig('train_scores.png')
    plt.show()
    
def test_tpot_MLP(test_rounds):
    scores = []
    current_score = 100
    train_scores = []
    print("classifiers: ", classifiers)    
    for i in range(20, 100, 20):  
        print(i)
        Tp = TPOTRegressor(max_time_mins =i)
        Tp.fit(ytrain_est, ytrain)
        scores.append(np.mean(abs(Tp.predict(yval_est)-yval)))
        train_scores.append(np.mean((abs(Tp.predict(ytrain_est)-ytrain))))
        current_score = np.mean(abs(Tp.predict(yval_est)-yval))
        if current_score == np.min(scores):
            print("score = ", current_score)
            print("scores are: ", scores)
            print("better tpot model")
            Tp.export('tpot_exported_pipeline2.py')
    plt.subplot(121)
    plt.title('valuation scores')
    plt.plot(scores)
    plt.savefig('Plots/eval_scores_mlp.png')
    plt.subplot(122)
    plt.plot(train_scores)
    plt.title('training scores')
    plt.savefig('Plots/train_scores_mlp.png')
    plt.show()
      
    
def use_tpot_as_MLP():
    exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=3, min_samples_split=20, n_estimators=100)
)
    exported_pipeline.fit(yval_est, yval)
    results = exported_pipeline.predict(predictions)
    return results, exported_pipeline
#%%
# plot feature importance
#mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
#ts = pd.Series(clf.booster().get_fscore())
#ts.index = ts.reset_index()['index'].map(mapFeat)
#ts.order()[-15:].plot(kind="barh", title=("features importance"))

#%%
# classify all loose predictors:
def classify_predictors():
    mseval = np.zeros((len(yval),len(classifiers)))
    msetrain = np.zeros((len(ytrain),len(classifiers)))
    for i in range(len(classifiers)):
        ytrain_est[ytrain_est[:,i]<0,i] = 0
        yval_est[yval_est[:,i]<0,i] = 0
        err = np.mean(abs(ytrain_est[:,i] - ytrain))
        print('Train error for {} is: {:.4f}'.format(classifiers[i],err))
        err = np.mean(abs(yval_est[:,i] - yval))
        print('Test error for {} is: {:.4f}'.format(classifiers[i],err))
        mseval[:,i] = abs(yval_est[:,i] - yval)
        msetrain[:,i] = abs(ytrain_est[:,i] - ytrain)
    
        if(min(predictions[:,i])<0):
            predictions[predictions[:,i]<0,i] = 0
        if(np.any(np.isnan(predictions[:,i]))):
            print(classifiers[i])
        if(np.all(np.isfinite(predictions[:,i]))==0):
            print(classifiers[i])
            
    msevaldf = pd.DataFrame(mseval)
    msetraindf = pd.DataFrame(msetrain)
    print('In total, by selecting the optimal classifier the training MSE is {:.2f}'.format(msetraindf.min(axis=1).mean()))
    print('In total, by selecting the optimal classifier the validation MSE is {:2f}'.format(msevaldf.min(axis=1).mean()))


#%%
# the MLP used to combine the regressors

from sklearn.neural_network import MLPRegressor
def use_mlp():
    models = []
    accuracy = []
    models_sse = []
    sse = []
    score_i = []
    for i in range(1,40):
        model_j = []
        score_j = []
        sse_j = []
        for j in range(0,15):
            clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(i))
            predictie = clf.fit(yval_est, yval)
            model_j.append(clf)
            score_j.append(np.mean(abs(clf.predict(yval_est) - yval)))
        print("Layer {} test accuracy: {:.4f}".format(i,min(score_j)))
        
        models.append(model_j[np.argmin(score_j)])
        accuracy.append(min(score_j))
        
    model_acc = np.argmin(accuracy)
    print('Best number of hlayers test acc = {}'.format(model_acc+1)) 
    
    clf = models[model_acc]
    y_est = clf.predict(predictions)
    y_est[y_est<=0] = 0
    return y_est, clf
#%%

def printStuff():
    print("minimum: ",np.min(y_est))
    print("maximum: ",np.max(y_est))
    print("average: ",np.average(y_est))
    estimatie_train = regressor.predict(ytrain_est)
    verschil_train = np.mean(abs(estimatie_train - ytrain))
    
    estimatie_val = regressor.predict(yval_est)
    print(len(estimatie_val))
    verschil_val = abs((regressor.predict(yval_est) - yval))
    grootste_verschil = np.max(abs((regressor.predict(yval_est) - yval)))
    index_grootste_verschil = np.argmax(grootste_verschil)
    index_grootste_verschil = index_grootste_verschil+1
    
    print("minimum: ",min(estimatie_train))
    print("maximum: ",max(estimatie_train))
    print("average: ",np.average(estimatie_train))
    print("verschil y train = ", np.average(verschil_train))
    print("verschil_val = ", np.average(verschil_val))
    print("grootste_verschil_val = ", grootste_verschil)

    
#%%
def testTpot():
    max = []
    train_max = []
    index = []
    index_j = []
    for i in range(2,10,2):
        for j in range(2,10,2):
            print("i = ", i, " j = ", j)
            Tp = TPOTRegressor(generations=i, population_size=j, cv=5, n_jobs=-1, verbosity =3)
            Tp.fit(Xtrain, ytrain)
            print("val score: ", np.mean(abs(Tp.predict(Xval)-yval)))
            print("train_score: ", np.mean((abs(Tp.predict(Xtrain)-ytrain))))
            max.append(np.mean(abs(Tp.predict(Xval)-yval)))
            index.append(i)
            index_j.append(j)
            train_max.append(np.mean(abs(Tp.predict(Xval)-yval)))
    plt.subplot(121)
    plt.plot(index, max)
    plt.subplot(122)
    plt.plot(index_j, max)
    plt.subplot(212)
    plt.plot(index, train_max)
    plt.subplot(222)
    plt.plot(index_j, train_max)
    plt.show()
    predictions[:,12] = Tp.predict(Xtest)
    ytrain_est[:,12] = Tp.predict(Xtrain)
    yval_est[:,12] = Tp.predict(Xval)
    
        
    
    
#%%
testTpot()
'''mlp = True
test_tpot = True
submit = True
tests=20
if test_tpot:
    #test_tpot_performance(tests)
    test_tpot_MLP(tests)
else:
    print("no testing today")

    
if submit:
    print("using tpot")
    y_est, regressor = use_tpot_as_MLP()
    printStuff()
    print("making submission")
    submission = pd.DataFrame(index=Test.index,columns=['seg_id','time_to_failure'])
    submission['seg_id'] = Test['seg_id'].values
    submission['time_to_failure'] = y_est
    submission.to_csv('submission.csv',index=False)

else:
    print("done")
'''