#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:45:51 2018

@author: ansh
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fmnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fmnist.load_data()

class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-Boot']

x_train = x_train / 255
x_test = x_test / 255


#show sample from the dataset

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(x_train[i], cmap = plt.cm.binary)
#    plt.xlabel(class_names[y_train[i]])
    

    
model = keras.Sequential([
        keras.layers.Flatten(input_shape = (28,28)),
        keras.layers.Dense(200, activation = tf.nn.leaky_relu),
        keras.layers.Dropout(0.015),
        keras.layers.Dense(200, activation = tf.nn.leaky_relu),
        keras.layers.Dropout(0.015),
        keras.layers.Dense(10, activation = tf.nn.softmax)]
)
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss = keras.losses.sparse_categorical_crossentropy,
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 15, batch_size=512)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy: ", test_accuracy)
        
predictions = model.predict(x_test)


plt.figure(figsize=(10,10))
for i in range(150,175):
    plt.subplot(5,5,i-149)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
