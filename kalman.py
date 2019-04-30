# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:49:17 2019

@author: Lisa
"""
import pandas as pd
import numpy as np

reader = pd.read_csv("Data/train.csv",
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},
                    chunksize=150000)

i=0

meandf = pd.DataFrame(index=range(0,150000),columns=['mean','std','max','min','sum','abs_sum','diff'])
for df in reader:
   if(i%50==0):
       print(i)
   i=i+1
   if(len(df)==150000):
        meandf['mean'] = meandf['mean']+((1/4194)*df['acoustic_data'].mean())
        meandf['std'] = meandf['std']+((1/4194)*df['acoustic_data'].std())
        meandf['max'] = meandf['std']+((1/4194)*df['acoustic_data'].max())
        meandf['min'] = meandf['std']+((1/4194)*df['acoustic_data'].min())
        meandf['sum'] = meandf['std']+((1/4194)*df['acoustic_data'].sum())
        meandf['abs_sum'] =meandf['std']+((1/4194)* np.abs(df['acoustic_data']).sum())
        meandf['diff'] = meandf['std']+((1/4194)*df['acoustic_data'].diff().mean())
        
    
meandf.to_csv('kalmanmodel.csv',index=False)
#%%

def kalman(chunk, model):
    #    
    n_iter = len(chunk)
    sz = (n_iter,) # size of array
    # truth value (typo??? in example intro kalman paper at top of p. 13 calls this z)
    z = chunk['acoustic_data'] # observations 
    Q =  model[2] # process variance
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=0         # a posteri error estimate
    xhatminus=0 # a priori estimate of x
    Pminus=0   # a priori error estimate
    K=0       # gain or blending factor
    R = model[1] #variance of the model
    # intial guesses
    xhat[:5] = model[0] #model mean
    P = 1.0
    Pminus = P+Q  #static Q    
    K = Pminus/( Pminus+R ) #static R  
    P = (1-K)*Pminus
    T = z.diff().rolling(window=5).mean()
    for k in range(5,n_iter):
        # time update
        xhatminus = xhat[k-1]+T.iloc[k]     
        xhat[k] = xhatminus+K*(z.iloc[k]-xhatminus)

    return xhat
    
def kalman_f(reader):
    i = 0
    m = meandf
    mse = []
    model = [m.mean()[0], m.std()[0], m.diff().mean()[0]]
    for chunk in reader:
        if(i%50==10):
            break
        x = kalman(chunk, model)
        diff = np.mean(abs(x - chunk['acoustic_data'].values))
        mse.append(diff.mean())
        i = i+1
    
    
    return mse


