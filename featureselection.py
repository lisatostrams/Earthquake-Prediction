#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:44:19 2019

@author: lisatostrams
"""

Xtrain,Xval,ytrain,yval = model_selection.train_test_split(X,y,test_size=0.5)
#dtc = tree.DecisionTreeRegressor(max_depth=3)

rf = RandomForestRegressor(n_estimators = d,max_depth=5)

#reg = Ridge(alpha=2)

#knn = KNeighborsRegressor(n_neighbors=45,algorithm='ball_tree')
#svmnorm = SVR(tol=d,gamma='auto')
dtc = rf.fit(Xtrain_norm, ytrain)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain_norm = scaler.transform(Xtrain)
Xval_norm = scaler.transform(Xval)
d_train = xgb.DMatrix(data=Xtrain_norm, label=ytrain, feature_names=Xtrain.columns)
d_val = xgb.DMatrix(data=Xval_norm, label=yval, feature_names=Xval.columns)
evallist = [(d_val, 'eval')]



err = []
xgb_params = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'verbosity': 0,
    'max_depth':3,
    'n_estimators':300,
    'silent':1,
    'gamma':1,
    'colsample_bytree':0.7,
    'min_child_weight':300
    
}

evallist = [(d_val, 'eval')]

#dtc = xgb.train(dtrain=d_train, num_boost_round=2000, evals=evallist, early_stopping_rounds=50,  params=xgb_params)


#%%


attributeNames = list(X)
levels = range(1,len(attributeNames))
error = np.zeros((2,len(levels)))


attribute_importance = [(attributeNames[i],dtc.feature_importances_[i]) for i in range(len(attributeNames))]
attributes_sorted = sorted(attribute_importance, key=lambda item: abs(item[1]), reverse=True)


print('Features in order of importance:')
print(*['{}: {:.4f}'.format(i[0],i[1]) for i in attributes_sorted],sep='\n')
err=[]
for t in levels:
    attributes = [i[0] for i in attributes_sorted][:t]
    scaler = preprocessing.StandardScaler().fit(Xtrain[attributes])
    Xtrain_norm = scaler.transform(Xtrain[attributes])
    Xval_norm = scaler.transform(Xval[attributes])
    d_train = xgb.DMatrix(data=Xtrain_norm, label=ytrain, feature_names=Xtrain[attributes].columns)
    d_val = xgb.DMatrix(data=Xval_norm, label=yval, feature_names=Xval[attributes].columns)
    evallist = [(d_val, 'eval')]

    dtc = xgb.train(dtrain=d_train, num_boost_round=2000, evals=evallist, early_stopping_rounds=50,  params=xgb_params)
    y_est = dtc.predict(d_val)
    mae = np.mean(abs(y_est-yval))
    evallist = [(d_train, 'eval')]
    dtc = xgb.train(dtrain=d_val, num_boost_round=2000, evals=evallist, early_stopping_rounds=50,  params=xgb_params)
    y_est = dtc.predict(d_train)
    mae += np.mean(abs(y_est-ytrain))
    print('dtc depth: level {} mae: {}'.format(t,mae/2))
    err.append((t,mae/2))

err = np.array(err)
idx = np.argmin(err[:,1])
plt.plot(err[:,0],err[:,1])
plt.title('Lowest error at level {}'.format(err[idx,0]))

plt.show()  

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


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = [i[0] for i in attributes_sorted][:23]
y_pos = np.arange(len(objects))
performance = [i[1] for i in attributes_sorted][:23]
 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)

plt.title('Decrease in Gini index if split on attribute')
plt.tight_layout()
plt.savefig('barplot.png',dpi=300)
plt.show()