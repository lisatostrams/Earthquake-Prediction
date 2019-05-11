# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:27:41 2019

@author: Lisa
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.signal import hilbert
from scipy.signal import hann
from numpy import convolve

def kalman(chunk, model,window=15):
    #    
    n_iter = len(chunk)
    sz = (n_iter,) # size of array
    # truth value (typo??? in example intro kalman paper at top of p. 13 calls this z)
    z = chunk['acoustic_data'] # observations 
    Q =  0.01 # process variance
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=0         # a posteri error estimate
    xhatminus=0 # a priori estimate of x
    Pminus=0   # a priori error estimate
    K=0       # gain or blending factor
    R = model[1] #variance of the model
    # intial guesses
    xhat[:window] = z[:window] #model mean
    P = 1.0
    Pminus = P+Q  #static Q    
    K = Pminus/( Pminus+R ) #static R  
    P = (1-K)*Pminus
    T = z.diff().rolling(window=window).mean()
    
    #B = lfilter([a], [1, -b], A)
    
    for k in range(window,n_iter):
        # time update
        xhatminus = xhat[k-1]+T.iloc[k]     
        xhat[k] = xhatminus+K*(z.iloc[k]-xhatminus)

    return xhat
    

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]


    
def cross_corr( x,taus):
    '''
    cross correlation function: for each tau, and for each channel, the signal is shifted by tau,
    dot multiplied with the complete original signal, and summed along the axis corresponding to the channels 
    '''
    n_signals = 1
    OUT = np.zeros([len(taus),n_signals])
    Xtau = np.zeros_like(x)
    m = 0
    x = x/x.std()
    x = x - x.mean()
    for tau in taus:
        for first in range(n_signals):
            Xtau = x.shift(tau)
            Xtau = Xtau.fillna(0)
        for second in range(n_signals):
            OUT[m,second] =  np.dot(Xtau,x)/len(x)
        m=m+1
            
    return OUT

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

def classic_sta_lta(x, length_sta, length_lta):
    
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

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


meanmodel = pd.read_csv('kalmanmodel.csv')
model = [meanmodel['mean'][0],meanmodel['std'][0],meanmodel['diff'][0]]


#%%
submission = pd.read_csv('Data/sample_submission.csv')
i=0
dtiny = 1e-5
Test = pd.DataFrame(index=range(len(submission)),columns=summary+['seg_id'])

pd.DataFrame().to_csv('XTest.csv',header=False)
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
    zc = np.fft.fft(x)
    
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    Test.loc[i, 'Rmean'] = realFFT.mean()
    Test.loc[i, 'Rstd'] = realFFT.std()
    Test.loc[i, 'Rmax'] = realFFT.max()
    Test.loc[i, 'Rmin'] = realFFT.min()
    Test.loc[i, 'Imean'] = imagFFT.mean()
    Test.loc[i, 'Istd'] = imagFFT.std()
    Test.loc[i, 'Imax'] = imagFFT.max()
    Test.loc[i, 'Imin'] = imagFFT.min()
    Test.loc[i, 'Rmean_last_5000'] = realFFT[-5000:].mean()
    Test.loc[i, 'Rstd__last_5000'] = realFFT[-5000:].std()
    Test.loc[i, 'Rmax_last_5000'] = realFFT[-5000:].max()
    Test.loc[i, 'Rmin_last_5000'] = realFFT[-5000:].min()
    Test.loc[i, 'Rmean_last_15000'] = realFFT[-15000:].mean()
    Test.loc[i, 'Rstd_last_15000'] = realFFT[-15000:].std()
    Test.loc[i, 'Rmax_last_15000'] = realFFT[-15000:].max()
    Test.loc[i, 'Rmin_last_15000'] = realFFT[-15000:].min()
    
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
    
    kalmanx = kalman(test, model,window=100)
    Test.loc[i, 'Kalman_correction'] = np.mean(abs(kalmanx - test['acoustic_data'].values))
    
    Test.loc[i, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    Test.loc[i, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
    Test.loc[i, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
    Test.loc[i, 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
    no_of_std = 3
    Test.loc[i, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
    Test.loc[i,'MA_700MA_BB_high_mean'] = (Test.loc[i, 'Moving_average_700_mean'] + no_of_std * Test.loc[i, 'MA_700MA_std_mean']).mean()
    Test.loc[i,'MA_700MA_BB_low_mean'] = (Test.loc[i, 'Moving_average_700_mean'] - no_of_std * Test.loc[i, 'MA_700MA_std_mean']).mean()
    Test.loc[i, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
    Test.loc[i,'MA_400MA_BB_high_mean'] = (Test.loc[i, 'Moving_average_700_mean'] + no_of_std * Test.loc[i, 'MA_400MA_std_mean']).mean()
    Test.loc[i,'MA_400MA_BB_low_mean'] = (Test.loc[i, 'Moving_average_700_mean'] - no_of_std * Test.loc[i, 'MA_400MA_std_mean']).mean()
    Test.loc[i, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
    Test.drop('Moving_average_700_mean', axis=1, inplace=True)
    
    Test.loc[i, 'q999'] = np.quantile(x,0.999)
    Test.loc[i, 'q001'] = np.quantile(x,0.001)
    
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
        Test.loc[i,'autocorr_'+str(lag)] = autoc[j][0]
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
    
    Test.loc[:i].to_csv('Xtest.csv',index=False)
    

#%%    
    
Test.to_csv('Xtest.csv',index=False)
