# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:27:33 2019

@author: Lisa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


reader = pd.read_csv("Data/train.csv",
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},
                    chunksize=150000)

summary= 'meanAudio stdAudio maxAudio maxTime minAudio minTime q75Audio q25Audio event n'.split()
summarized_data = np.zeros((4195,len(summary)))
i = 0
start_time = time.time()
for df in reader:
    mean = df.mean()
    std = df.std()
    maxi = df.max()
    mini = df.min()
    q75 = df.quantile(.75)
    q25 = df.quantile(.25)
    summarized_data[i,0] = mean[0]
    summarized_data[i,1] = std[0]
    summarized_data[i,2] = maxi[0]
    summarized_data[i,3] = maxi[1]
    summarized_data[i,4] = mini[0]
    summarized_data[i,5] = mini[1]
    summarized_data[i,6] = q75[0]
    summarized_data[i,7] = q25[0]

    summarized_data[i,9] = len(df)
    i=i+1
    if(i%500==0):
        print(i)
summarized_data = pd.DataFrame(summarized_data,columns=summary)
min0 = summarized_data['minTime'].min()
summarized_data['event'] = (summarized_data['minTime'].diff()>2).astype(int)
events = summarized_data.index[summarized_data['event'] == 1]
summarized_data.to_csv('summarized_data_150000.csv')
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
reader = pd.read_csv("Data/train.csv",
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},
                    chunksize=150000)
events_df = []
i = 0
for df in reader:
    if(i in events):
        print(i)
        events_df.append(df)
    i=i+1
    if(i%500==0):
        print(i)
    
#%%
i = 0
for df in events_df:
    df.to_csv('event_{}.csv'.format(i))
    i=i+1

#%%
    
fig, ax = plt.subplots(4,4,figsize=(24,18))
i=0
for k in range(0,4):
    for j in range(0,4):
        #events_df[i]['acoustic_data'].plot(ax=ax[k,j])
        ax2 = ax[k,j].twinx()
        events_df[i]['time_to_failure'].plot(color='orange',ax=ax2)
        i=i+1
        

        
    