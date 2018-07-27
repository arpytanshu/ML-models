#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:30:31 2018

@author: ansh
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import shutil


from sklearn.datasets import load_boston
#directory for model checkpoint
#OUTDIR = "./modeldirectory_new"

boston = load_boston()

train_data = boston.data[0:450,:]
test_data = boston.data[450: , :]

train_label = boston.target[0:450]
test_label = boston.target[450:]

#set logging to INFO < more verbose than default >
tf.logging.set_verbosity(tf.logging.INFO)

#START AFRESH EVERY TIME
#shutil.rmtree(OUTDIR, ignore_errors = True)

def train_input_fn():
    features = {}
    for i, item in enumerate(boston.feature_names):
        add = {str(item) : train_data[:, i]}
        features.update(add)
    labels = train_label
    return tf.estimator.inputs.numpy_input_fn(
            features, labels,
            batch_size = 64,
            num_epochs = 10,
            shuffle = True)

def eval_input_fn():
    features = {}
    for i, item in enumerate(boston.feature_names):
        add = {str(item) : test_data[:, i]}
        features.update(add)
    labels = test_label
    return tf.estimator.inputs.numpy_input_fn(
            x = features,
            y = labels,
            shuffle=True)


def make_feature_cols():
    input_columns = [tf.feature_column.numeric_column(k) for k in boston.feature_names]
    return input_columns

#model_LinearRegressor = tf.estimator.LinearRegressor(feature_columns = make_feature_cols(),
#                                                     model_dir = OUTDIR)

model_DNNRegressor = tf.estimator.DNNRegressor(hidden_units = [50],
                                               feature_columns = make_feature_cols(),
#                                               model_dir = OUTDIR, 
                                               activation_fn = tf.nn.relu,
                                               dropout = 0.01)

#model_LinearRegressor.train(input_fn = train_input_fn(), steps = 1000)
model_DNNRegressor.train(input_fn = train_input_fn(), steps = 10000)
model_DNNRegressor.evaluate(input_fn = eval_input_fn(), steps=1)