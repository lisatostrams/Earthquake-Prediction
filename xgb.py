# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:31:12 2019

@author: Lisa
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:20:11 2019

@author: Lisa
"""
summary= attributes
X = chunks[summary]
y = chunks['minTime']


X,Xtest,y,ytest = model_selection.train_test_split(X,y,test_size=0.5)

#%%
import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective="reg:linear",max_depth=1)

xgb_model = xgb_model.fit(X, y.values)


y_est = xgb_model.predict(Xtest)
y_est[y_est<0] = 0
import sklearn.metrics as metric
print('MSE xgb: {:.4f}'.format(metric.mean_squared_error(ytest,y_est)))
#%%


#X = chunks[summary]
#y = chunks['minTime']
y_est = xgb_model.predict(X)
y_est[y_est<0] = 0

plt.style.use('ggplot')
fig, ax = plt.subplots(10,1,figsize=(16,30))
i=0
coeff = xgb_model.feature_importances_
attribute_importance = [(summary[i],coeff[i]) for i in range(len(summary))]
attributes_sorted = sorted(attribute_importance, key=lambda item: abs(item[1]), reverse=True)
plots = [s[0] for s in attributes_sorted[:10]]
coeff = [s[1] for s in attributes_sorted]
for s in plots:
    ax2 = ax[i].twinx()
    chunks['endTime'].plot(color=list(plt.rcParams['axes.prop_cycle'])[1]['color'],alpha=0.8,ax=ax[i])
    ax[i].grid(False)
    ax[i].tick_params(axis='y',labelleft=False,left=False)
    #ax[i,0].set_ylim([0.1,1])
    chunks[s].plot(ax = ax2,alpha=.95)
    ax2.set_ylabel('Time to failure',color=list(plt.rcParams['axes.prop_cycle'])[1]['color'])
    lim = ax[i].get_ylim()
    ax3 = ax[i].twinx()
    ax3.plot(y_est,color='g',alpha=0.6)
    ax3.set_ylim(lim)
    ax3.tick_params(axis='y',labelleft=False,left=False)
    ax3.grid(False)
    ax[i].set_title('xgb feature importance: {:.4f}'.format(coeff[i]))
    ax[i].set_ylabel(s,fontsize=16)
    i=i+1
    
plt.tight_layout()
plt.savefig('xgb_prediction.png',dpi=300)


#%%


submission = pd.read_csv('Data/sample_submission.csv')

for file in submission['seg_id']:
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
    #X_tr['abs_q05'] = np.quantile(np.abs(x), 0.05)
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
        X_tr[ 'av_change_rate_roll_mean_' + str(windows)] = 0#np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X_tr[ 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
    
    
    X_tr = X_tr.drop('abs_max_roll_mean_1000',axis=1)
    X_tr = X_tr[summary]
    y_est = xgb_model.predict(X_tr)
    y_est[y_est<0] = 0
    submission.loc[submission['seg_id']==file,'time_to_failure'] = y_est[0]
    
#%%
    
submission.to_csv('submissionxgb.csv',index=False)
