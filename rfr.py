# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:46:52 2019

@author: Lisa
"""

 
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
summary= 'meanAudio medianAudio modeAudio stdAudio maxAudio minAudio q75Audio q25Audio'.split()
X = chunks[summary]
y = chunks['minTime']

import sklearn.model_selection as model_selection

X,Xtest,y,ytest = model_selection.train_test_split(X,y,test_size=0.5)

#%%
from sklearn.ensemble import RandomForestRegressor

rfr_model = RandomForestRegressor(n_estimators=200)

rfr_model = rfr_model.fit(X, y)


y_est = rfr_model.predict(Xtest)
y_est[y_est<0] = 0
import sklearn.metrics as metric
print('r2 Score linear regression: {:.4f}'.format(metric.mean_squared_error(ytest,y_est)))
#%%


X = chunks[summary]
y = chunks['minTime']
y_est = rfr_model.predict(X)
y_est[y_est<0] = 0

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
    
plt.savefig('rfr_prediction.png',dpi=300)


#%%


submission = pd.read_csv('Data/sample_submission.csv')

for file in submission['seg_id']:
    test = pd.read_csv('Test/{}.csv'.format(file))
    summary_test = np.zeros((1,len(summary)))
    mean = test.mean()
    median = test.median()
    mode = test['acoustic_data'].mode()
    std = test.std()
    maxi = test.max()
    mini = test.min()
    q75 = test.quantile(.75)
    q25 = test.quantile(.25)
    summary_test[0,0] = mean[0]
    summary_test[0,1] = median[0]
    summary_test[0,2] = mode[0]
    summary_test[0,3] = std[0]
    summary_test[0,4] = maxi[0]
    summary_test[0,5] = mini[0]
    summary_test[0,6] = q75[0]
    summary_test[0,7] = q25[0]
    summarized_data = pd.DataFrame(summary_test,columns=summary)
    y_est = rfr.predict(summarized_data)
    y_est[y_est<0] = 0
    submission.loc[submission['seg_id']==file,'time_to_failure'] = y_est[0]
    
#%%
    
submission.to_csv('submissionrfr.csv',index=False)
