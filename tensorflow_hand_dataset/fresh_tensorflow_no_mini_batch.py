#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 00:00:51 2017

@author: ansh
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math, h5py



def load_dataset():
    train_data = h5py.File("train_signs.h5")
    test_data = h5py.File("test_signs.h5")
    
    X_train = train_data["train_set_x"][:]
    Y_train = train_data["train_set_y"][:]
    X_test = test_data["test_set_x"][:]
    Y_test = test_data["test_set_y"][:]
    Y_train = tf.one_hot(indices = Y_train, depth = 6, axis = 0)
    Y_test = tf.one_hot(indices = Y_test, depth = 6, axis = 0)
    with tf.Session() as sess:
        Y_train = sess.run(Y_train)
        Y_test = sess.run(Y_test)
        #   unrolling X's
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T
    
    return X_train, Y_train, X_test, Y_test
    

#X_train, Y_train, X_test, Y_test = load_dataset()