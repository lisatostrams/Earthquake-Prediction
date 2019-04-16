# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:27:33 2019

@author: Lisa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

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

reader = pd.read_csv("Data/train.csv",
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},
                    chunksize=150000)

summ= ('meanAudio medianAudio modeAudio stdAudio stdAudioIncrease ' +
            'maxAudio maxTime minAudio endTime minTime q75Audio q25Audio event ' +
            'mean_change_abs mean_change_rate').split()
col = ['std_first_50000', 'std_last_50000', 'std_first_10000',
'std_last_10000', 'avg_first_50000', 'avg_last_50000',
'avg_first_10000', 'avg_last_10000', 'min_first_50000',
'min_last_50000', 'min_first_10000', 'min_last_10000',
'max_first_50000', 'max_last_50000', 'max_first_10000',
'max_last_10000', 'max_to_min', 'max_to_min_diff', 'count_big',
'sum', 'mean_change_rate_first_50000',
'mean_change_rate_last_50000', 'mean_change_rate_first_10000',
'mean_change_rate_last_10000', 'q95', 'q99', 'q05', 'q01',
'abs_q95', 'abs_q99', 'abs_q05', 'abs_q01', 'trend', 'abs_trend',
'abs_mean', 'abs_std', 'mad', 'kurt', 'skew', 'ave_roll_std_10',
'std_roll_std_10', 'max_roll_std_10', 'min_roll_std_10',
'q01_roll_std_10', 'q05_roll_std_10', 'q95_roll_std_10',
'q99_roll_std_10', 'av_change_abs_roll_std_10',
'av_change_rate_roll_std_10', 'abs_max_roll_std_10',
'ave_roll_mean_10', 'std_roll_mean_10', 'max_roll_mean_10',
'min_roll_mean_10', 'q01_roll_mean_10', 'q05_roll_mean_10',
'q95_roll_mean_10', 'q99_roll_mean_10',
'av_change_abs_roll_mean_10', 'av_change_rate_roll_mean_10',
'abs_max_roll_mean_10', 'ave_roll_std_100', 'std_roll_std_100',
'max_roll_std_100', 'min_roll_std_100', 'q01_roll_std_100',
'q05_roll_std_100', 'q95_roll_std_100', 'q99_roll_std_100',
'av_change_abs_roll_std_100', 'av_change_rate_roll_std_100',
'abs_max_roll_std_100', 'ave_roll_mean_100', 'std_roll_mean_100',
'max_roll_mean_100', 'min_roll_mean_100', 'q01_roll_mean_100',
'q05_roll_mean_100', 'q95_roll_mean_100', 'q99_roll_mean_100',
'av_change_abs_roll_mean_100', 'av_change_rate_roll_mean_100',
'abs_max_roll_mean_100', 'ave_roll_std_1000', 'std_roll_std_1000',
'max_roll_std_1000', 'min_roll_std_1000', 'q01_roll_std_1000',
'q05_roll_std_1000', 'q95_roll_std_1000', 'q99_roll_std_1000',
'av_change_abs_roll_std_1000', 'av_change_rate_roll_std_1000',
'abs_max_roll_std_1000', 'ave_roll_mean_1000',
'std_roll_mean_1000', 'max_roll_mean_1000', 'min_roll_mean_1000',
'q01_roll_mean_1000', 'q05_roll_mean_1000', 'q95_roll_mean_1000',
'q99_roll_mean_1000', 'av_change_abs_roll_mean_1000',
'av_change_rate_roll_mean_1000', 'abs_max_roll_mean_1000']
summary = summ+col
summarized_data = np.zeros((4195,len(summary)))
i = 0
start_time = time.time()
for df in reader:
    maxi = df.max()
    mini = df.min()
    q75 = df.quantile(.75)
    q25 = df.quantile(.25)
    summarized_data[i,0] = df.mean()[0]
    summarized_data[i,1] = df.median()[0]
    summarized_data[i,2] = df['acoustic_data'].mode()[0]
    summarized_data[i,3] = df.std()[0]
    summarized_data[i,4] = np.mean((df.rolling(1000).std()< df.rolling(10000).std()))[0]
    summarized_data[i,5] = maxi[0]
    summarized_data[i,6] = maxi[1]
    summarized_data[i,7] = mini[0]
    summarized_data[i,8] = df.iloc[-1]['time_to_failure']
    summarized_data[i,9] = mini[1]
    summarized_data[i,10] = q75[0]
    summarized_data[i,11] = q25[0]
    x = pd.Series(df['acoustic_data'].values)
    summarized_data[i,12] = np.mean(np.diff(x))
    summarized_data[i,13] = calc_change_rate(x)
    X_tr = pd.DataFrame(index=range(0,1))
    X_tr['std_first_50000'] = x[:50000].std()
    X_tr['std_last_50000'] = x[-50000:].std()
    X_tr['std_first_10000'] = x[:10000].std()
    X_tr['std_last_10000'] = x[-10000:].std()
    
    X_tr['avg_first_50000'] = x[:50000].mean()
    X_tr['avg_last_50000'] = x[-50000:].mean()
    X_tr['avg_first_10000'] = x[:10000].mean()
    X_tr['avg_last_10000'] = x[-10000:].mean()
    
    X_tr['min_first_50000'] = x[:50000].min()
    X_tr['min_last_50000'] = x[-50000:].min()
    X_tr['min_first_10000'] = x[:10000].min()
    X_tr['min_last_10000'] = x[-10000:].min()
    
    X_tr['max_first_50000'] = x[:50000].max()
    X_tr['max_last_50000'] = x[-50000:].max()
    X_tr['max_first_10000'] = x[:10000].max()
    X_tr['max_last_10000'] = x[-10000:].max()
    
    X_tr['max_to_min'] = x.max() / np.abs(x.min())
    X_tr['max_to_min_diff'] = x.max() - np.abs(x.min())
    X_tr['count_big'] = len(x[np.abs(x) > 500])
    X_tr['sum'] = x.sum()
    
    X_tr['mean_change_rate_first_50000'] = calc_change_rate(x[:50000])
    X_tr['mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])
    X_tr['mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
    X_tr['mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

    X_tr['q95'] = np.quantile(x, 0.95)
    X_tr['q99'] = np.quantile(x, 0.99)
    X_tr['q05'] = np.quantile(x, 0.05)
    X_tr['q01'] = np.quantile(x, 0.01)
    
    X_tr['abs_q95'] = np.quantile(np.abs(x), 0.95)
    X_tr['abs_q99'] = np.quantile(np.abs(x), 0.99)
    X_tr['abs_q05'] = np.quantile(np.abs(x), 0.05)
    X_tr['abs_q01'] = np.quantile(np.abs(x), 0.01)

    X_tr['trend'] = add_trend_feature(x)
    X_tr['abs_trend'] = add_trend_feature(x, abs_values=True)
    X_tr['abs_mean'] = np.abs(x).mean()
    X_tr['abs_std'] = np.abs(x).std()
    
    X_tr['mad'] = x.mad()
    X_tr['kurt'] = x.kurtosis()
    X_tr['skew'] = x.skew()
    for windows in [10, 100, 1000]:
        x_roll_std = x.rolling(windows).std().dropna().values
        x_roll_mean = x.rolling(windows).mean().dropna().values
        
        X_tr[ 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X_tr[ 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X_tr[ 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X_tr[ 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X_tr[ 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X_tr[ 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X_tr[ 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X_tr[ 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X_tr[ 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X_tr[ 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X_tr[ 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
        X_tr[ 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X_tr[ 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X_tr[ 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X_tr[ 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X_tr[ 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X_tr[ 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X_tr[ 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X_tr[ 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X_tr[ 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X_tr[ 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X_tr[ 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
    summarized_data[i,14:-1] = X_tr.values
    i=i+1
    if(i%500==0):
        print(i)
summarized_data = pd.DataFrame(summarized_data,columns=summary)
min0 = summarized_data['minTime'].min()
summarized_data['event'] = (summarized_data['minTime'].diff()>2).astype(int)
events = summarized_data.index[summarized_data['event'] == 1]
summarized_data.to_csv('summarized_data_150000.csv',index=False)
print('This took {:.2f} seconds to process'.format(time.time() - start_time))
#train_acoustic_data_small = train['acoustic_data'].values[::50]
#train_time_to_failure_small = train['time_to_failure'].values[::50]
#
#fig, ax1 = plt.subplots(figsize=(16, 8))
#plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")
#plt.plot(train_acoustic_data_small, color='b')
#ax1.set_ylabel('acoustic_data', color='b')
#plt.legend(['acoustic_data'])
#ax2 = ax1.twinx()
#plt.plot(train_time_to_failure_small, color='g')
#ax2.set_ylabel('time_to_failure', color='g')
#plt.legend(['time_to_failure'], loc=(0.875, 0.9))
#plt.grid(False)
#
#del train_acoustic_data_small
#del train_time_to_failure_small

#%%

#%%
#reader = pd.read_csv("Data/train.csv",
#                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},
#                    chunksize=150000)
#events_df = []
#i = 0
#for df in reader:
#    if(i+1 in events):
#        print(i)
#        events_df.append(df)
#    i=i+1
#    if(i%500==0):
#        print(i)
#    
##%%
#i = 0
#for df in events_df:
#    df.to_csv('event_{}.csv'.format(i))
#    i=i+1

#%%
    
#fig, ax = plt.subplots(4,4,figsize=(24,18))
#i=0
#for k in range(0,4):
#    for j in range(0,4):
#        ax2 = ax[k,j].twinx()
#        events_df[i]['time_to_failure'].plot(color='orange',ax=ax2,alpha=0.7)
#        events_df[i]['acoustic_data'].plot(ax=ax[k,j])        
#        i=i+1
#        
#plt.savefig('events.png',dpi=300)
#        
    