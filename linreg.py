# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:20:11 2019

@author: Lisa
"""
#summary= attributes
X = chunks[summary]
y = chunks['endTime']

X,Xtest,y,ytest = model_selection.train_test_split(X,y,test_size=0.5)


#%%
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X, y)

reg.score(X, y)

y_est = reg.predict(Xtest)
y_est[y_est<0] = 0
import sklearn.metrics as metric
print('mse linear regression: {:.4f}'.format(metric.mean_squared_error(ytest,y_est)))
#%%

#X = chunks[summary]
#y = chunks['endTime']
y_est = reg.predict(X)
#y_est[y_est<0] = 0
#X = X.fillna(0)

plt.style.use('ggplot')

i=0
coeff = reg.coef_
attribute_importance = [(summary[i],coeff[i]) for i in range(len(summary))]
attributes_sorted = sorted(attribute_importance, key=lambda item: abs(item[1]), reverse=True)
plots = [s[0] for s in attributes_sorted[:10]]
coeff = [s[1] for s in attributes_sorted]
fig, ax = plt.subplots(10,1,figsize=(16,24))
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
    ax[i].set_ylabel(s,fontsize=10)
    ax[i].set_title('linreg coefficient: {:.4f}'.format(coeff[i]))
    i=i+1
    
plt.tight_layout()
plt.savefig('linreg_prediction.png',dpi=300)


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
    y_est = reg.predict(summarized_data)
    y_est[y_est<0] = 0
    submission.loc[submission['seg_id']==file,'time_to_failure'] = y_est[0]
    
#%%
    
submission.to_csv('submissionlinreg.csv',index=False)
