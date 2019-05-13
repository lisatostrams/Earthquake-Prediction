# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:05:09 2019

@author: Lisa
"""

max_depth = 2

n_estimators = 500

tol = 0.001


xgb_params = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'verbosity': 0,
    'max_depth':2,
}



#%%
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

Xtrain,Xval,ytrain,yval = model_selection.train_test_split(X,y,test_size=0.4)

Test = pd.read_csv('Xtest.csv')
Xtest = Test[summary]

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain_norm = scaler.transform(Xtrain)
Xval_norm = scaler.transform(Xval)
Xtest_norm = scaler.transform(Xtest)

#%% train all classifiers
        
classifiers = 'DTC RF LINREG KNN SVM SVMlinear BRR HR XGB ADA GP LGBM TPOT'.split(sep=' ')
predictions = np.zeros((len(Xtest),len(classifiers)))
ytrain_est = np.zeros((len(Xtrain),len(classifiers)))
yval_est = np.zeros((len(Xval),len(classifiers)))

dtc = tree.DecisionTreeRegressor(max_depth=max_depth) #train decision tree
dtc = dtc.fit(Xtrain,ytrain)
predictions[:,0] = dtc.predict(Xtest)
ytrain_est[:,0] = dtc.predict(Xtrain)
yval_est[:,0] = dtc.predict(Xval)

rf = RandomForestRegressor(n_estimators = n_estimators)
rf = rf.fit(Xtrain, ytrain)
predictions[:,1] = rf.predict(Xtest)
ytrain_est[:,1] = rf.predict(Xtrain)
yval_est[:,1] = rf.predict(Xval)

reg = LinearRegression()
reg = reg.fit(Xtrain, ytrain)
predictions[:,2] = reg.predict(Xtest)
ytrain_est[:,2] = reg.predict(Xtrain)
yval_est[:,2] = reg.predict(Xval)

knn = KNeighborsRegressor(n_neighbors=25,algorithm='ball_tree')
knn = knn.fit(Xtrain,ytrain)
predictions[:,3] = knn.predict(Xtest)
ytrain_est[:,3] = knn.predict(Xtrain)
yval_est[:,3] = knn.predict(Xval)

svmnorm = SVR(tol=tol,gamma='auto')
svmnorm = svmnorm.fit(Xtrain_norm, ytrain)
predictions[:,4] = svmnorm.predict(Xtest_norm)
ytrain_est[:,4] = svmnorm.predict(Xtrain_norm)
yval_est[:,4] = svmnorm.predict(Xval_norm)

svmlnorm = LinearSVR(max_iter=5000)
svmlnorm = svmlnorm.fit(Xtrain_norm,ytrain)
predictions[:,5] = svmlnorm.predict(Xtest_norm)
ytrain_est[:,5] = svmlnorm.predict(Xtrain_norm)
yval_est[:,5] = svmlnorm.predict(Xval_norm)

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

# eval
d_train = xgb.DMatrix(data=Xtrain_norm, label=ytrain, feature_names=Xtrain.columns)
d_val = xgb.DMatrix(data=Xval_norm, label=yval, feature_names=Xval.columns)
evallist = [(d_val, 'eval'), (d_train, 'train')]
model = xgb.train(dtrain=d_train, num_boost_round=100, evals=evallist, early_stopping_rounds=50,  params=xgb_params)
predictions[:,8] = model.predict(xgb.DMatrix(data=Xtest_norm, feature_names=Xtest.columns))
ytrain_est[:,8] = model.predict(d_train, ntree_limit=model.best_ntree_limit)
yval_est[:,8] = model.predict(d_val, ntree_limit=model.best_ntree_limit)

#mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
#ts = pd.Series(clf.booster().get_fscore())
#ts.index = ts.reset_index()['index'].map(mapFeat)
#ts.order()[-15:].plot(kind="barh", title=("features importance"))

abc = AdaBoostRegressor()
abc = abc.fit(Xtrain,ytrain)
predictions[:,9] = abc.predict(Xtest)
ytrain_est[:,9] = abc.predict(Xtrain)
yval_est[:,9] = abc.predict(Xval)

kernel = np.var(y)* RBF(length_scale=1)
gp = GaussianProcessRegressor(kernel=kernel,alpha=0.1).fit(Xtrain_norm, ytrain)
predictions[:,10] = gp.predict(Xtest_norm)
ytrain_est[:,10] = gp.predict(Xtrain_norm)
yval_est[:,10] = gp.predict(Xval_norm)

lgb_train = lgb.Dataset(Xtrain, ytrain)
lgb_eval = lgb.Dataset(Xval, yval, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
print("train lgb")
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=60000,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')
print('Starting prediction...')
# predict
predictions[:,11] = gbm.predict(Xtest, num_iteration=gbm.best_iteration)
ytrain_est[:,11] = gbm.predict(Xtrain, num_iteration=gbm.best_iteration)
yval_est[:,11] = gbm.predict(Xval, num_iteration = gbm.best_iteration)


#%%

from tpot import TPOTRegressor


print(Xtrain.shape)
print(ytrain.shape)
Tp = ExtraTreesRegressor(bootstrap=True, max_features=0.55, min_samples_leaf=2, min_samples_split=15, n_estimators=100)
Tp.fit(Xtrain,ytrain)
predictions[:,12] = Tp.predict(Xtest)
ytrain_est[:,12] = Tp.predict(Xtrain)
yval_est[:,12] = Tp.predict(Xval)

#%%
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
# generating the best TPOT model as mlp:
'''
from tpot import TPOTRegressor

pipeline_optimizer = TPOTRegressor()
pipeline_optimizer.fit(yval_est, yval)
print('execute optimizer')
print(pipeline_optimizer.score(yval_est, yval))
pipeline_optimizer.export('tpot_exported_pipeline.py')
'''
#%%
# generating the best tpot model as feature:
'''
from tpot import TPOTRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

print(Xtrain.shape)
print(ytrain.shape)
#pipeline_optimizer = TPOTRegressor(verbosity = 2)
Tp = make_pipeline(
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=1.0, loss="linear", n_estimators=100)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=11, min_samples_split=18, n_estimators=100)
)
Tp.fit(Xtrain_norm, ytrain)
predictions[:,12] = Tp.predict(Xtest_norm)
ytrain_est[:,12] = Tp.predict(Xtrain_norm)
yval_est[:,12] = Tp.predict(Xval_norm)

'''
#%%
# using optimal pipeline for predictions as total
'''
from tpot.builtins import OneHotEncoder
from sklearn.linear_model import LassoLarsCV

exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    LassoLarsCV(normalize=False)
)

exported_pipeline.fit(yval_est, yval)
results = exported_pipeline.predict(predictions)
'''

#%%
# use TPOT as an extra regressor for the MLP:

#%%
# the MLP used to bring all code together
from sklearn.neural_network import MLPRegressor
models = []
accuracy = []
models_sse = []
sse = []
for i in range(1,30):
    model_j = []
    score_j = []
    sse_j = []
    for j in range(0,10):
        clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(i,))
        clf.fit(yval_est, yval)
        model_j.append(clf)
        score_j.append(np.mean(abs(clf.predict(yval_est) - yval)))
        
    
    print("Layer {} test accuracy: {:.4f}".format(i,min(score_j)))
    print()
    
    models.append(model_j[np.argmin(score_j)])
    accuracy.append(min(score_j))
#%%
model_acc = np.argmin(accuracy)
print('Best number of hlayers test acc = {}'.format(model_acc+1)) 

clf = models[model_acc]
y_est = clf.predict(predictions)
y_est[y_est<0] = 0
#%%

print("making submission")
submission = pd.DataFrame(index=Test.index,columns=['seg_id','time_to_failure'])
submission['seg_id'] = Test['seg_id'].values
submission['time_to_failure'] = y_est
submission.to_csv('submission.csv',index=False)