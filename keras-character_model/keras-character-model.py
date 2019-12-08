#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 20:39:11 2018

@author: ansh
"""

from __future__ import print_function
import string
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random, sys, io

#file for logging output

with open("./nietzsche.txt", encoding='utf-8') as f:
    text = f.read().lower()
#remove everything except ascii_letters, '.', ',' & ' ' 
text = ''.join(c for c in text if c in string.ascii_letters+"., ")


chars = sorted(list(set(text)))

char_index = {}
index_char = {}

#make char->int & int->char mapping
for i, char in enumerate(chars):
    char_index[char] = i
    index_char[i] = char
    
#make sentences of fixed length
#put the first character after the sentence as prediction for training
step = 3            # make a new sentence of every 'step'th character
maxlen = 40         #length of sentence
sentences = []      #list of sentence
next_chars = []     #list of next character after each sentence

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), np.bool)


for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_index[char]] = 1
    y[i, char_index[next_chars[i]]] = 1
    
# build model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

model = keras.Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.001)

model.compile(optimizer, loss = 'categorical_crossentropy')


def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# define callback
def on_epoch_end(epoch, logs):
    
    tempstr = '==== Generating Text after Epoch: ' + str(epoch) + ' ====\n'
    print(tempstr)
    
    
    start_index = random.randint(0, len(text) - maxlen - 1)
    
    for diversity in [0.25, 1.0, 1.25]:
        tempstr = "== diversity: " + str(diversity)+ " ==\n"
        print(tempstr)
        
        generated = ''
        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        
        tempstr = 'generating with seed: " ' + sentence + ' "'
        print(tempstr)
    
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0,t,char_index[char]] = 1.
            
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = index_char[next_index]
            
            generated += next_char
            sentence = sentence[1:] + next_char
            
            sys.stdout.write(next_char)
            sys.stdout.flush()
            
        print()
        
print_callback = keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

# fit
model.fit(x, y,
          batch_size = 256,
          epochs = 60,
          callbacks=[print_callback])
            
    
