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

X = chunks[summary]
y = chunks['minTime']

#%%
import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective="reg:linear")

xgb_model = xgb_model.fit(X, y)


y_est = xgb_model.predict(X)
y_est[y_est<0] = 0
import sklearn.metrics as metric
print('r2 Score linear regression: {:.4f}'.format(metric.mean_squared_error(y,y_est)))
#%%

plt.style.use('ggplot')
fig, ax = plt.subplots(len(summary),1,figsize=(16,20))
i=0
for s in summary:
    ax2 = ax[i].twinx()
    chunks['minTime'].plot(color=list(plt.rcParams['axes.prop_cycle'])[1]['color'],alpha=0.8,ax=ax[i])
    ax[i].grid(False)
    ax[i].tick_params(axis='y',labelleft=False,left=False)
    #ax[i,0].set_ylim([0.1,1])
    chunks[s].plot(ax = ax2,alpha=1)
    ax2.set_ylabel('Time to failure',color=list(plt.rcParams['axes.prop_cycle'])[1]['color'])
    ax3 = ax[i].twinx()
    ax3.plot(y_est,color='g',alpha=0.8)
    ax3.tick_params(axis='y',labelleft=False,left=False)
    ax3.grid(False)
    ax[i].set_ylabel(s,fontsize=26)
    i=i+1
    
plt.savefig('xgb_prediction.png',dpi=300)


#%%


submission = pd.read_csv('Data/sample_submission.csv')

for file in submission['seg_id']:
    test = pd.read_csv('Test/{}.csv'.format(file))
    summary_test = np.zeros((1,len(summary)))
    mean = test.mean()
    std = test.std()
    maxi = test.max()
    mini = test.min()
    q75 = test.quantile(.75)
    q25 = test.quantile(.25)
    summary_test[0,0] = mean[0]
    summary_test[0,1] = std[0]
    summary_test[0,2] = maxi[0]
    summary_test[0,3] = mini[0]
    summary_test[0,4] = q75[0]
    summary_test[0,5] = q25[0]
    summarized_data = pd.DataFrame(summary_test,columns=summary)
    y_est = reg.predict(summarized_data)
    y_est[y_est<0] = 0
    submission.loc[submission['seg_id']==file,'time_to_failure'] = y_est[0]
    
#%%
    
submission.to_csv('submission.csv',index=False)
