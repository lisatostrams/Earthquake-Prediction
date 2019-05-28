#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:11:53 2019

@author: lisatostrams
"""

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from keras.optimizers import Adam
import pandas as pd
import numpy as np
timesteps = 150000
input_dim = 1

latent_dim = 1000

inputs = Input(shape=(timesteps//100, input_dim))
encoded = LSTM(latent_dim)(inputs)


decoded = RepeatVector(timesteps//100)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0000, amsgrad=False) #'rmsprop'
sequence_autoencoder.compile(optimizer=optimizer,loss='mae')
#sequence_autoencoder.load_weights('aemodels/model_trained_chunk_0.h5')
#1-voor-1 de latente variabelen veranderen en zien wat er veranderd in de reconstructie 

#%%

reader = pd.read_csv("Data/train.csv",
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},
                    chunksize=timesteps)

cols = ['latent_dim_{}'.format(i) for i in range(latent_dim)]
Train = pd.DataFrame(index=range(4195),columns=cols)
i=0
for df in reader:
    print(i)
    x_train = df['acoustic_data'].values.reshape(100,timesteps//100,input_dim)
    sequence_autoencoder.fit(x_train, x_train, epochs=30, batch_size=10, validation_data=(x_train, x_train))
    sequence_autoencoder.save('aemodels/model_trained_chunk_{}.h5'.format(i))
    i=i+1
    if(i==10):
        break
    
    #x_test = x_test.astype('float32') / 255.
    #x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    #x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
