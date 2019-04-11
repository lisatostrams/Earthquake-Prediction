# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:44:14 2019

@author: Lisa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
chunks = pd.read_csv("summarized_data_10000.csv")
print('There are {} chunks in the file.'.format(len(chunks)))

#%%

chunks_event = chunks[chunks['event']==1]
chunks_other = chunks[chunks['event']==0]
plt.style.use('ggplot')
summary= 'meanAudio stdAudio maxAudio minAudio q75Audio q25Audio'.split()
fig, ax = plt.subplots(len(summary),2,figsize=(16,20))
i=0
for s in summary:
    
    chunks_other[s].hist(color='g',ax = ax[i,0],bins=100)
    chunks_event[s].hist(color='r',ax= ax[i,1],bins=1)
    ax[i,0].set_ylabel(s,fontsize=26)
    i=i+1
    
ax[0,0].set_title('Histogram non events (100 bins)',fontsize=18)
ax[0,1].set_title('Histogram events (100 bins)',fontsize=18)
plt.tight_layout()
plt.savefig('summary.png',dpi=300)