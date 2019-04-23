# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:27:41 2019

@author: Lisa
"""
import warnings
warnings.filterwarnings("ignore")
submission = pd.read_csv('Data/sample_submission.csv')
i=0
Test = pd.DataFrame(index=submission['seg_id'],columns=summary)
for file in submission['seg_id']:
    if(i%50==0):
        print(i)
    i=i+1
    test = pd.read_csv('Test/{}.csv'.format(file))
    X_tr = pd.DataFrame(index=range(0,1))
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
    X_tr['meanAudio'] = mean[0]
    X_tr['medianAudio'] = median[0]
    X_tr['modeAudio'] = mode[0]
    X_tr['stdAudio'] = std[0]
    X_tr['stdAudioIncrease'] = np.mean((test.rolling(1000).std()< test.rolling(10000).std()))[0]
    X_tr['maxAudio'] = maxi[0]
    X_tr['minAudio'] = mini[0]
    X_tr['q75Audio'] = q75[0]
    X_tr['q25Audio'] = q25[0]
    x = pd.Series(test['acoustic_data'].values)
    X_tr['mean_change_abs'] = np.mean(np.diff(x))
    X_tr['mean_change_rate'] = calc_change_rate(x)
    
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
    
    
    Test.loc[Test.index==file,summary] = X_tr[summary].values

    
    
Test.to_csv('test.csv')
