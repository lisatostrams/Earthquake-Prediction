# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:32:11 2019

@author: Lisa
"""

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

def kalman(chunk, model):
    #    
    n_iter = len(chunk)
    sz = (n_iter,) # size of array
    # truth value (typo??? in example intro kalman paper at top of p. 13 calls this z)
    z = chunk['acoustic_data'] # observations 
    Q =  0.1 # process variance
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=0         # a posteri error estimate
    xhatminus=0 # a priori estimate of x
    Pminus=0   # a priori error estimate
    K=0       # gain or blending factor
    R = model[1] #variance of the model
    # intial guesses
    xhat[:5] = z[:5] #model mean
    P = 1.0
    Pminus = P+Q  #static Q    
    K = Pminus/( Pminus+R ) #static R  
    P = (1-K)*Pminus
    T = z.diff().rolling(window=5).mean()
    
    #B = lfilter([a], [1, -b], A)
    
    for k in range(5,n_iter):
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
              'trend','q99_roll_std_10','modeAudio','q99_roll_std_1000','medianAudio','max_to_min','abs_q01','q99_roll_std_100'
              'abs_q05', 'Hann_window_mean_50', 'Hann_window_mean_150','Hann_window_mean_1500',
              'Hann_window_mean_15000','classic_sta_lta1_mean','classic_sta_lta2_mean','classic_sta_lta3_mean',
              'classic_sta_lta4_mean','classic_sta_lta5_mean','classic_sta_lta6_mean','classic_sta_lta7_mean','classic_sta_lta8_mean','autocorr_1',
              'autocorr_5','autocorr_10','autocorr_50','autocorr_100','autocorr_500','autocorr_1000','autocorr_5000','autocorr_10000','abs_max_roll_mean_1000',
              'abs_q05','Kalman_correction','exp_Moving_average_300_mean','exp_Moving_average_3000_mean',
              'exp_Moving_average_30000_mean','MA_700MA_std_mean','MA_700MA_BB_high_mean','MA_700MA_BB_low_mean',
              'MA_400MA_std_mean','MA_400MA_BB_high_mean','MA_400MA_BB_low_mean','MA_1000MA_std_mean','q999','q001','q99_roll_std_100',
              'Rmean','Rstd','Rmax','Rmin','Imean','Istd','Imax','Imin','Rmean_last_5000','Rstd__last_5000','Rmax_last_5000','Rmin_last_5000',
              'Rmean_last_15000','Rstd_last_15000','Rmax_last_15000','Rmin_last_15000']


meanmodel = pd.read_csv('kalmanmodel.csv')
model = [meanmodel['mean'][0],meanmodel['std'][0],meanmodel['diff'][0]]
#%%
reader = pd.read_csv("Data/train.csv",
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},
                    chunksize=150000)
Train = pd.DataFrame(index=range(4195),columns=summary+['endTime'])
dtiny = 1e-5



#%%

#%%
import time
summarized_data = np.zeros((4195,len(summary)))
i = 0
start_time = time.time()
for df in reader:
    if(i%50==1):
        break
        print(i)
    maxi = df.max()
    mini = df.min()
    q75 = df.quantile(.75)
    q25 = df.quantile(.25)
    mean = df.mean()
    median = df.median()
    mode = df['acoustic_data'].mode()
    std = df.std()
    maxi = df.max()
    mini = df.min()
    q75 = df.quantile(.75)
    q25 = df.quantile(.25)
    Train.loc[i,'endTime'] = df.iloc[-1]['time_to_failure']
    Train.loc[i,'meanAudio'] = mean[0]
    Train.loc[i,'medianAudio'] = median[0]
    Train.loc[i,'modeAudio'] = mode[0]
    Train.loc[i,'stdAudio'] = std[0]
    Train.loc[i,'stdAudioIncrease'] = np.mean((df.rolling(1000).std()< df.rolling(10000).std()))[0]
    Train.loc[i,'maxAudio'] = maxi[0]
    Train.loc[i,'minAudio'] = mini[0]
    Train.loc[i,'q75Audio'] = q75[0]
    Train.loc[i,'q25Audio'] = q25[0]
    x = pd.Series(df['acoustic_data'].values)
    zc = np.fft.fft(x)
    
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    Train.loc[i, 'Rmean'] = realFFT.mean()
    Train.loc[i, 'Rstd'] = realFFT.std()
    Train.loc[i, 'Rmax'] = realFFT.max()
    Train.loc[i, 'Rmin'] = realFFT.min()
    Train.loc[i, 'Imean'] = imagFFT.mean()
    Train.loc[i, 'Istd'] = imagFFT.std()
    Train.loc[i, 'Imax'] = imagFFT.max()
    Train.loc[i, 'Imin'] = imagFFT.min()
    Train.loc[i, 'Rmean_last_5000'] = realFFT[-5000:].mean()
    Train.loc[i, 'Rstd__last_5000'] = realFFT[-5000:].std()
    Train.loc[i, 'Rmax_last_5000'] = realFFT[-5000:].max()
    Train.loc[i, 'Rmin_last_5000'] = realFFT[-5000:].min()
    Train.loc[i, 'Rmean_last_15000'] = realFFT[-15000:].mean()
    Train.loc[i, 'Rstd_last_15000'] = realFFT[-15000:].std()
    Train.loc[i, 'Rmax_last_15000'] = realFFT[-15000:].max()
    Train.loc[i, 'Rmin_last_15000'] = realFFT[-15000:].min()
    
    Train.loc[i,'mean_change_abs'] = np.mean(np.diff(x))
    Train.loc[i,'mean_change_rate'] = calc_change_rate(x)
    
    Train.loc[i,'std_first_50000'] = x[:50000].std()
    Train.loc[i,'std_last_50000'] = x[-50000:].std()
    Train.loc[i,'std_first_10000'] = x[:10000].std()
    Train.loc[i,'std_last_10000'] = x[-10000:].std()
    
    Train.loc[i,'avg_first_50000'] = x[:50000].mean()
    Train.loc[i,'avg_last_50000'] = x[-50000:].mean()
    Train.loc[i,'avg_first_10000'] = x[:10000].mean()
    Train.loc[i,'avg_last_10000'] = x[-10000:].mean()
    
    Train.loc[i,'min_first_50000'] = x[:50000].min()
    Train.loc[i,'min_last_50000'] = x[-50000:].min()
    Train.loc[i,'min_first_10000'] = x[:10000].min()
    Train.loc[i,'min_last_10000'] = x[-10000:].min()
    
    Train.loc[i,'max_first_50000'] = x[:50000].max()
    Train.loc[i,'max_last_50000'] = x[-50000:].max()
    Train.loc[i,'max_first_10000'] = x[:10000].max()
    Train.loc[i,'max_last_10000'] = x[-10000:].max()
    
    Train.loc[i,'max_to_min'] = x.max() / np.abs(x.min())
    Train.loc[i,'max_to_min_diff'] = x.max() - np.abs(x.min())
    Train.loc[i,'count_big'] = len(x[np.abs(x) > 500])
    Train.loc[i,'sum'] = x.sum()
    
    Train.loc[i,'mean_change_rate_first_50000'] = calc_change_rate(x[:50000])
    Train.loc[i,'mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])
    Train.loc[i,'mean_change_rate_first_10000'] = calc_change_rate(x[:10000])
    Train.loc[i,'mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

    Train.loc[i,'q95'] = np.quantile(x, 0.95)
    Train.loc[i,'q99'] = np.quantile(x, 0.99)
    Train.loc[i,'q05'] = np.quantile(x, 0.05)
    Train.loc[i,'q01'] = np.quantile(x, 0.01)
    
    Train.loc[i,'abs_q95'] = np.quantile(np.abs(x), 0.95)
    Train.loc[i,'abs_q99'] = np.quantile(np.abs(x), 0.99)
    Train.loc[i,'abs_q05'] = np.quantile(np.abs(x), 0.05)
    Train.loc[i,'abs_q01'] = np.quantile(np.abs(x), 0.01)

    Train.loc[i,'trend'] = add_trend_feature(x)
    Train.loc[i,'abs_trend'] = add_trend_feature(x, abs_values=True)
    Train.loc[i,'abs_mean'] = np.abs(x).mean()
    Train.loc[i,'abs_std'] = np.abs(x).std()
    
    Train.loc[i,'mad'] = x.mad()
    Train.loc[i,'kurt'] = x.kurtosis()
    Train.loc[i,'skew'] = x.skew()
    
    Train.loc[i,'hmean'] = stats.hmean(np.abs(x[np.nonzero(x)[0]]))
    Train.loc[i,'gmean'] = stats.gmean(np.abs(x[np.nonzero(x)[0]]))
    
    Train.loc[i,'Hilbert_mean'] = np.abs(hilbert(x)).mean()
    
    kalmanx = kalman(df, model)
    Train.loc[i, 'Kalman_correction'] = np.mean(abs(kalmanx - df['acoustic_data'].values))
    
    Train.loc[i, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    Train.loc[i, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
    Train.loc[i, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
    Train.loc[i, 'exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)
    no_of_std = 3
    Train.loc[i, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
    Train.loc[i,'MA_700MA_BB_high_mean'] = (Train.loc[i, 'Moving_average_700_mean'] + no_of_std * Train.loc[i, 'MA_700MA_std_mean']).mean()
    Train.loc[i,'MA_700MA_BB_low_mean'] = (Train.loc[i, 'Moving_average_700_mean'] - no_of_std * Train.loc[i, 'MA_700MA_std_mean']).mean()
    Train.loc[i, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
    Train.loc[i,'MA_400MA_BB_high_mean'] = (Train.loc[i, 'Moving_average_700_mean'] + no_of_std * Train.loc[i, 'MA_400MA_std_mean']).mean()
    Train.loc[i,'MA_400MA_BB_low_mean'] = (Train.loc[i, 'Moving_average_700_mean'] - no_of_std * Train.loc[i, 'MA_400MA_std_mean']).mean()
    Train.loc[i, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
    Train.drop('Moving_average_700_mean', axis=1, inplace=True)
    
    Train.loc[i, 'q999'] = np.quantile(x,0.999)
    Train.loc[i, 'q001'] = np.quantile(x,0.001)
    
    Train.loc[i,'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
    Train.loc[i,'iqr1'] = np.subtract(*np.percentile(x, [95, 5]))
    Train.loc[i,'ave10'] = stats.trim_mean(x, 0.1)
    
    
    hann_windows = [50, 150, 1500, 15000]
    for hw in hann_windows:
        Train.loc[i,f'Hann_window_mean_'+str(hw)] = (convolve(x, hann(hw), mode='same') / sum(hann(hw))).mean()

    Train.loc[i,'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
    Train.loc[i,'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
    Train.loc[i,'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
    Train.loc[i,'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
    Train.loc[i,'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
    Train.loc[i,'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
    Train.loc[i,'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
    Train.loc[i,'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
    
    autocorr_lags = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    autoc = cross_corr(x,autocorr_lags)
    j=0
    for lag in autocorr_lags:
        Train.loc[i,'autocorr_'+str(lag)] = autoc[j]
        j=j+1
    
    
    
    for windows in [10, 100, 1000]:
        x_roll_std = x.rolling(windows).std().dropna().values
        x_roll_mean = x.rolling(windows).mean().dropna().values
        
        idx = x_roll_mean == 0
        x_roll_mean[idx] = dtiny
        
        Train.loc[i, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        Train.loc[i, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        Train.loc[i, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        Train.loc[i, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        Train.loc[i, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        Train.loc[i, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        Train.loc[i, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        Train.loc[i, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        Train.loc[i, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        Train.loc[i, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        Train.loc[i, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
        Train.loc[i, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        Train.loc[i, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        Train.loc[i, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        Train.loc[i, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        Train.loc[i, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        Train.loc[i, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        Train.loc[i, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        Train.loc[i, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        Train.loc[i, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        Train.loc[i, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        Train.loc[i, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
    i=i+1

#%%    
    
#Train.to_csv('XTrain.csv')
