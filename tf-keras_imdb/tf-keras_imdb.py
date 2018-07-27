#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 01:59:55 2018

@author: ansh
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras


max_features = 200000   # vocabulary size
maxlen = 80             #length of each review
batch_size = 1024
embedding_size = 128
#load data from keras.datasets
imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

num_train_samples = len(x_train)
num_test_samples = len(x_test)
print(num_train_samples, 'train sequences')
print(num_test_samples, 'test sequences')

#load word to index dict
word_index = imdb.get_word_index()
#make index to word dict
index_word = {v: k for k, v in word_index.items()}



print('sample sequences from dataset')
def print_samples(A=[]):
    assert len(A) <= num_train_samples
    for j in A:
        for i in x_train[j]:
            print(index_word[i], end=' ')
        print('\n')


#padding sequences (samples X time)
#x_train = numpy array of lists (the lists are of variable length)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen = maxlen)



#Build Model

model = keras.Sequential()
model.add(keras.layers.Embedding(max_features, embedding_size))
model.add(keras.layers.LSTM(embedding_size, dropout=0.1, return_sequences= True ,recurrent_dropout=0.2))
model.add(keras.layers.LSTM(embedding_size, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()


model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

print('Train. . .')

model.fit(x_train, y_train, batch_size=batch_size,
          epochs = 15,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score: ', score)
print('Test accuracy: ', acc)




































