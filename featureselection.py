#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:44:19 2019

@author: lisatostrams
"""

Xtrain,Xval,ytrain,yval = model_selection.train_test_split(X,y,test_size=0.5)
#dtc = tree.DecisionTreeRegressor(max_depth=3)

rf = RandomForestRegressor(n_estimators = d,max_depth=5)
dtc = rf.fit(Xtrain, ytrain)


#%%


attributeNames = list(X)
levels = range(1,len(attributeNames))
error = np.zeros((2,len(levels)))


attribute_importance = [(attributeNames[i],dtc.feature_importances_[i]) for i in range(len(attributeNames))]
attributes_sorted = sorted(attribute_importance, key=lambda item: item[1], reverse=True)


print('Features in order of importance:')
print(*['{}: {:.4f}'.format(i[0],i[1]) for i in attributes_sorted],sep='\n')
err=[]
for t in levels:
    attributes = [i[0] for i in attributes_sorted][:t]
    dtc = tree.DecisionTreeRegressor(max_depth=3)
    dtc = dtc.fit(Xtrain[attributes],ytrain)
    y_est = dtc.predict(Xval[attributes])
    mae = np.mean(abs(y_est-yval))
    dtc = dtc.fit(Xval[attributes],yval)
    y_est = dtc.predict(Xtrain[attributes])
    mae += np.mean(abs(y_est-ytrain))
    print('dtc depth: level {} mae: {}'.format(t,mae/2))
    err.append((t,mae/2))

err = np.array(err)
idx = np.argmin(err[:,1])
plt.plot(err[:,0],err[:,1])
plt.title('Lowest error at level {}'.format(err[idx,0]))

plt.show()  

dtc_attributes = ['q05_roll_std_10', 'q05_roll_std_100']
rf_attributes = ['q05_roll_std_100']


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