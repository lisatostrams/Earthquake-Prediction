# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:27:33 2019

@author: Lisa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("Data/train.csv",
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float16})

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
time_0 = train['time_to_failure'].min()
events = (train['time_to_failure']==time_0).astype(int)
n_events = sum(events.diff())
#%%

    