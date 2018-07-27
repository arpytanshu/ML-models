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
from tensorflow.python.framework import ops

from PIL import Image
from scipy import ndimage
import scipy


n_x = 12288
n_hl1 = 25
n_hl2 = 12
n_y = 6
nn_arch = [n_x,n_hl1,n_hl2,n_y]

#n_x = 12288
#n_hl1 = 100
#n_hl2 = 45
#n_hl3 = 15
#n_y = 6
#nn_arch = [n_x,n_hl1,n_hl2,n_hl3,n_y]


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
    

def random_mini_batches(X, Y, minibatch_size = 512):
    '''
    shape of X: n x m (input size X num of samples)
    call this on one  hotted Y's
    '''
    m = X.shape[1]
    minibatches = []
    
    #shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    
    num_complete_minibatches = math.floor(m / minibatch_size)
    for k in range(0, num_complete_minibatches):
        minibatch_X = shuffled_X[:, k*minibatch_size : (k+1)*minibatch_size]
        minibatch_Y = shuffled_Y[:, k*minibatch_size : (k+1)*minibatch_size]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)
        
    if m % minibatch_size != 0:
        minibatch_X = shuffled_X[:, num_complete_minibatches*minibatch_size : m]
        minibatch_Y = shuffled_Y[:, num_complete_minibatches*minibatch_size : m]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)
    
    return minibatches


#def initialize_parameters():
#    W1 = tf.get_variable("W1", [25, 12288], tf.float32, tf.contrib.layers.xavier_initializer())
#    W2 = tf.get_variable("W2", [12, 25], tf.float32, tf.contrib.layers.xavier_initializer())
#    W3 = tf.get_variable("W3", [6, 12], tf.float32, tf.contrib.layers.xavier_initializer())
#    b1 = tf.get_variable("b1", [25, 1], tf.float32, tf.zeros_initializer())
#    b2 = tf.get_variable("b2", [12, 1], tf.float32, tf.zeros_initializer())
#    b3 = tf.get_variable("b3", [6, 1], tf.float32, tf.zeros_initializer())
#    parameters = {"W1" : W1,
#                  "W2" : W2,
#                  "W3" : W3,
#                  "b1" : b1,
#                  "b2" : b2,
#                  "b3" : b3
#                  }
#    return parameters


def initialize_parameters(nn_arch = nn_arch):
    n = len(nn_arch)
    parameters = {}
    for i in range(1,n):
        parameters["W{0}".format(i)] = tf.get_variable("W"+str(i), (nn_arch[i], nn_arch[i-1]), dtype = tf.float32,
                                                       initializer = tf.contrib.layers.xavier_initializer())
        parameters["b{0}".format(i)] = tf.get_variable("b"+str(i), (nn_arch[i], 1), dtype = tf.float32,
                                                       initializer = tf.zeros_initializer())
    return parameters

def compute_cost(Z_last, Y):
    # Z_last is the logit output(that goes for activation) of the last "hidden" layer
    logits = tf.transpose(Z_last)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

def compute_cost_w_regu(Z_last, Y, m, parameters, lambd=0.1):
    
    # Z_last is the logit output(that goes for activation) of the last "hidden" layer
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    regu_term = (lambd / m) * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
    
    logits = tf.transpose(Z_last)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels) + regu_term)
    return cost

def create_placeholders(n_x, n_y):
    X = tf.placeholder(shape=(n_x, None), dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape=(n_y, None), dtype = tf.float32, name = 'Y')
    return X, Y
    
def forward_propagation(X, parameters):
    #unpack parameters
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3


#def forward_propagation(X, parameters):
#    #unpack parameters
#    W1 = parameters["W1"] # 100 x 12288
#    W2 = parameters["W2"] # 45 x 100
#    W3 = parameters["W3"] # 15 x 45
#    W4 = parameters["W4"] # 6 x 15
#    
#    b1 = parameters["b1"] # 100 x 1
#    b2 = parameters["b2"] # 45 x 1
#    b3 = parameters["b3"] # 15 x 1
#    b4 = parameters["b4"] # 6 x 1
#    
#    Z1 = tf.add(tf.matmul(W1, X), b1)
#    A1 = tf.nn.relu(Z1)
#    Z2 = tf.add(tf.matmul(W2, A1), b2)
#    A2 = tf.nn.relu(Z2)
#    Z3 = tf.add(tf.matmul(W3, A2), b3)
#    A3 = tf.nn.relu(Z3)
#    Z4 = tf.add(tf.matmul(W4, A3), b4)
#
#    return Z4


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 512, print_cost = True):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
   
   
    X, Y = create_placeholders(n_x, n_y)
    
    parameters = initialize_parameters()
    Z_last = forward_propagation(X, parameters)
    cost = compute_cost_w_regu(Z_last, Y, m, parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m/minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size = minibatch_size)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z_last), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


def sigmoid(z):
    return 1 / (1 + tf.exp(-z))
    
def predict(X, parameters):
    
    z_last = forward_propagation(X, parameters)
    y_hat = sigmoid(z_last)
    return y_hat
    
    
    
X_train, Y_train, X_test, Y_test = load_dataset()
X_train = X_train / 255.
X_test = X_test / 255.

parameters = model(X_train, Y_train, X_test, Y_test)


