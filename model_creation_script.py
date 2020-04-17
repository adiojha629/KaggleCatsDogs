# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:23:28 2020

@author: adioj
"""

import tensorflow as tf

#Network from scratch

#Input layer
inp = tf.keras.layers.Input(shape = (128,128,3))

#First convolutional layer has 32 filters with 3x3 kernels, Relu activation
conv1 = tf.keras.layers.Conv2D(32,kernel_size = (3, 3), activation = 'relu', padding = 'same')(inp)

pool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2))(conv1)

out = tf.keras.layers.Dense(2,activation = 'softmax')(pool1)

model = tf.keras.Model(inputs = inp, outputs = out)

