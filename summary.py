# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:44:14 2019

@author: Lisa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
chunks = pd.read_csv("XTrain.csv")
print('There are {} chunks in the file.'.format(len(chunks)))
timediff = chunks['endTime'].diff()


#%%

chunks_event = chunks.iloc[events]
bad_df = chunks.index.isin(events)
chunks_other = chunks[~bad_df]

#%%
corrf = []
for s in summary:
    if(not 'time' in s.lower() and not 'event' in s.lower()):
        c = chunks[s].corr(chunks['endTime'])
        if(np.isfinite(c)):
            corrf.append((s,c))
attributes_sorted = sorted(corrf, key=lambda item: abs(item[1]), reverse=True)
attributes = [s[0] for s in attributes_sorted]

#%%
plt.style.use('ggplot')

sumplot = [s[0] for s in attributes_sorted[:40]]
fig, ax = plt.subplots(10,4,figsize=(40,40))
i=0
j=0
for s in sumplot:
    
    
    
    ax2 = ax[i,j].twinx()
    
    chunks['endTime'].plot(color=list(plt.rcParams['axes.prop_cycle'])[1]['color'],alpha=0.8,ax=ax2)
    chunks[s].plot(ax = ax[i,j],alpha=.95)
    #ax[i,j].grid(False)
    ax2.grid(False)
    ax[i,j].set_ylabel(s,fontsize=15,color='r')
    ax[i,j].set_title('{} Correlation with endTime: {:.4f}'.format(s,chunks[s].corr(chunks['endTime'])))
    j=j+1
    if(j%4==0):
        i=i+1
        j=0
    

plt.tight_layout()
plt.savefig('summary.png',dpi=300)

#%%
chunks_before_event = chunks.iloc[events-1]
bad_df = chunks.index.isin(list(events) + (list(events-1)))
chunks_other = chunks[~bad_df]
#%%
plt.style.use('ggplot')
summary= 'meanAudio medianAudio modeAudio stdAudio stdAudioIncrease maxAudio minAudio endTime q75Audio q25Audio'.split()
fig, ax = plt.subplots(len(sumplot),1,figsize=(8,50))
i=0
for s in sumplot:
    
    chunks_other[s].hist(color='g',ax = ax[i],bins=100,alpha=.7)
    ax2 = ax[i].twinx()
    chunks_event[s].hist(color='r',ax= ax2,bins=16,grid=False,alpha=.7)
    ax3 = ax[i].twinx()
    ax3.set_yticks([])
    chunks_before_event[s].hist(color='b',ax=ax3,bins=16,grid=False,alpha=.6)
    ax[i].set_ylabel(s,fontsize=20)
    i=i+1

plt.tight_layout()
plt.savefig('summary3.png',dpi=300)