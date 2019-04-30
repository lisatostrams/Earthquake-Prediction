# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:27:41 2019

@author: Lisa
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calc_change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

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
              'trend','q99_roll_std_10','modeAudio','q99_roll_std_1000','medianAudio','max_to_min','abs_q01','q99_roll_std_100'
              'abs_q05', 'Hann_window_mean_50', 'Hann_window_mean_150','Hann_window_mean_1500',
              'Hann_window_mean_15000','classic_sta_lta1_mean','classic_sta_lta2_mean','classic_sta_lta3_mean',
              'classic_sta_lta4_mean','classic_sta_lta5_mean','classic_sta_lta6_mean','classic_sta_lta7_mean','classic_sta_lta8_mean','autocorr_1',
              'autocorr_5','autocorr_10','autocorr_50','autocorr_100','autocorr_500',
              'autocorr_1000','autocorr_5000','autocorr_10000','abs_max_roll_mean_1000']

#%%
submission = pd.read_csv('Data/sample_submission.csv')
i=0
dtiny = 1e-5
Test = pd.DataFrame(index=range(len(submission)),columns=summary+['seg_id'])
#%%

for file in submission['seg_id']:
    if(i%50==0):
        print(i)
    
    test = pd.read_csv('Test/{}.csv'.format(file))
    maxi = test.max()
    mini = test.min()
    q75 = test.quantile(.75)
    q25 = test.quantile(.25)
    mean = test.mean()
    median = test.median()
    mode = test['acoustic_data'].mode()
    std = test.std()
    maxi = test.max()
    mini = test.min()
    q75 = test.quantile(.75)
    q25 = test.quantile(.25)
    Test.loc[i,'seg_id'] = file
    Test.loc[i,'meanAudio'] = mean[0]
    Test.loc[i,'medianAudio'] = median[0]
    Test.loc[i,'modeAudio'] = mode[0]
    Test.loc[i,'stdAudio'] = std[0]
    Test.loc[i,'stdAudioIncrease'] = np.mean((test.rolling(1000).std()< test.rolling(10000).std()))[0]
    Test.loc[i,'maxAudio'] = maxi[0]
    Test.loc[i,'minAudio'] = mini[0]
    Test.loc[i,'q75Audio'] = q75[0]
    Test.loc[i,'q25Audio'] = q25[0]
    x = pd.Series(test['acoustic_data'].values)
    Test.loc[i,'mean_change_abs'] = np.mean(np.diff(x))
    Test.loc[i,'mean_change_rate'] = calc_change_rate(x)
    
    Test.loc[i,'std_first_50000'] = x[:50000].std()
    Test.loc[i,'std_last_50000'] = x[-50000:].std()
    Test.loc[i,'std_first_10000'] = x[:10000].std()
    Test.loc[i,'std_last_10000'] = x[-10000:].std()
    
    Test.loc[i,'avg_first_50000'] = x[:50000].mean()
    Test.loc[i,'avg_last_50000'] = x[-50000:].mean()
    Test.loc[i,'avg_first_10000'] = x[:10000].mean()
    Test.loc[i,'avg_last_10000'] = x[-10000:].mean()
    
    Test.loc[i,'min_first_50000'] = x[:50000].min()
    Test.loc[i,'min_last_50000'] = x[-50000:].min()
    Test.loc[i,'min_first_10000'] = x[:10000].min()
    Test.loc[i,'min_last_10000'] = x[-10000:].min()
    
    Test.loc[i,'max_first_50000'] = x[:50000].max()
    Test.loc[i,'max_last_50000'] = x[-50000:].max()
    Test.loc[i,'max_first_10000'] = x[:10000].max()
    Test.loc[i,'max_last_10000'] = x[-10000:].max()
    
    Test.loc[i,'max_to_min'] = x.max() / np.abs(x.min())
    Test.loc[i,'max_to_min_diff'] = x.max() - np.abs(x.min())
    Test.loc[i,'count_big'] = len(x[np.abs(x) > 500])
    Test.loc[i,'sum'] = x.sum()
    
    Test.loc[i,'mean_change_rate_first_50000'] = calc_change_rate(x[:50000])
    Test.loc[i,'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])
    Test.loc[i,'mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
    Test.loc[i,'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

    Test.loc[i,'q95'] = np.quantile(x, 0.95)
    Test.loc[i,'q99'] = np.quantile(x, 0.99)
    Test.loc[i,'q05'] = np.quantile(x, 0.05)
    Test.loc[i,'q01'] = np.quantile(x, 0.01)
    
    Test.loc[i,'abs_q95'] = np.quantile(np.abs(x), 0.95)
    Test.loc[i,'abs_q99'] = np.quantile(np.abs(x), 0.99)
    Test.loc[i,'abs_q05'] = np.quantile(np.abs(x), 0.05)
    Test.loc[i,'abs_q01'] = np.quantile(np.abs(x), 0.01)

    Test.loc[i,'trend'] = add_trend_feature(x)
    Test.loc[i,'abs_trend'] = add_trend_feature(x, abs_values=True)
    Test.loc[i,'abs_mean'] = np.abs(x).mean()
    Test.loc[i,'abs_std'] = np.abs(x).std()
    
    Test.loc[i,'mad'] = x.mad()
    Test.loc[i,'kurt'] = x.kurtosis()
    Test.loc[i,'skew'] = x.skew()
    
    Test.loc[i,'hmean'] = stats.hmean(np.abs(x[np.nonzero(x)[0]]))
    Test.loc[i,'gmean'] = stats.gmean(np.abs(x[np.nonzero(x)[0]]))
    
    Test.loc[i,'Hilbert_mean'] = np.abs(hilbert(x)).mean()
    
    Test.loc[i,'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
    Test.loc[i,'iqr1'] = np.subtract(*np.percentile(x, [95, 5]))
    Test.loc[i,'ave10'] = stats.trim_mean(x, 0.1)
    
    hann_windows = [50, 150, 1500, 15000]
    for hw in hann_windows:
        Test.loc[i,f'Hann_window_mean_'+str(hw)] = (convolve(x, hann(hw), mode='same') / sum(hann(hw))).mean()

    Test.loc[i,'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
    Test.loc[i,'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
    Test.loc[i,'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
    Test.loc[i,'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
    Test.loc[i,'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
    Test.loc[i,'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
    Test.loc[i,'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
    Test.loc[i,'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
    
    autocorr_lags = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    autoc = cross_corr(x,autocorr_lags)
    j=0
    for lag in autocorr_lags:
        Test.loc[i,'autocorr_'+str(lag)] = autoc[j]
        j=j+1
    
    
    for windows in [10, 100, 1000]:
        x_roll_std = x.rolling(windows).std().dropna().values
        x_roll_mean = x.rolling(windows).mean().dropna().values
        
        idx = x_roll_mean == 0
        x_roll_mean[idx] = dtiny
        
        Test.loc[i, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        Test.loc[i, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        Test.loc[i, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        Test.loc[i, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        Test.loc[i, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        Test.loc[i, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        Test.loc[i, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        Test.loc[i, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        Test.loc[i, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        Test.loc[i, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        Test.loc[i, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
        Test.loc[i, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        Test.loc[i, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        Test.loc[i, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        Test.loc[i, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        Test.loc[i, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        Test.loc[i, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        Test.loc[i, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        Test.loc[i, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        Test.loc[i, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        Test.loc[i, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        Test.loc[i, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
    i=i+1
    

#%%    
    
Test.to_csv('Xtest.csv')
