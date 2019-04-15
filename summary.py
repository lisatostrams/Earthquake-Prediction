# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:44:14 2019

@author: Lisa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
chunks = pd.read_csv("summarized_data_150000.csv")
print('There are {} chunks in the file.'.format(len(chunks)))
events = chunks.index[chunks['event'] == 1]-1

#%%

chunks_event = chunks.iloc[events]
bad_df = chunks.index.isin(events)
chunks_other = chunks[~bad_df]
#%%
plt.style.use('ggplot')
summary= 'meanAudio medianAudio modeAudio stdAudio maxAudio minAudio endTime q75Audio q25Audio'.split()
fig, ax = plt.subplots(len(summary),1,figsize=(8,24))
i=0
for s in summary:
    
    chunks_other[s].hist(color='g',ax = ax[i],bins=100,alpha=.8)

    ax2 = ax[i].twinx()

    chunks_event[s].hist(color='r',ax= ax2,bins=16,alpha=.7,grid=False)
    ax[i].set_ylabel(s,fontsize=26)
    i=i+1

plt.tight_layout()
plt.savefig('summary2.png',dpi=300)

#%%
chunks_before_event = chunks.iloc[events-1]
bad_df = chunks.index.isin(list(events) + (list(events-1)))
chunks_other = chunks[~bad_df]
#%%
plt.style.use('ggplot')
summary= 'meanAudio stdAudio maxAudio minAudio endTime q75Audio q25Audio'.split()
fig, ax = plt.subplots(len(summary),1,figsize=(8,20))
i=0
for s in summary:
    
    chunks_other[s].hist(color='g',ax = ax[i],bins=100,alpha=.7)
    ax2 = ax[i].twinx()
    chunks_event[s].hist(color='r',ax= ax2,bins=16,grid=False,alpha=.7)
    ax3 = ax[i].twinx()
    ax3.set_yticks([])
    chunks_before_event[s].hist(color='b',ax=ax3,bins=16,grid=False,alpha=.6)
    ax[i].set_ylabel(s,fontsize=26)
    i=i+1

plt.tight_layout()
plt.savefig('summary3.png',dpi=300)