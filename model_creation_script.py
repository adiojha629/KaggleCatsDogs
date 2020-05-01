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

conv2 = tf.keras.layers.Conv2D(32,kernel_size = (3, 3), activation = 'relu', padding = 'same')(pool1)

pool2 = tf.keras.layers.MaxPool2D(pool_size = (2,2))(conv2)

flat = tf.keras.layers.Flatten()(pool2)

hidden1 = tf.keras.layers.Dense(256, activation = 'relu')(flat)

dropout1 = tf.keras.layers.Dropout(rate = 0.3)(hidden1)

hidden2 = tf.keras.layers.Dense(256, activation = 'relu')(dropout1)

dropout2 = tf.keras.layers.Dropout(rate = 0.3)(hidden2)

out = tf.keras.layers.Dense(2, activation = 'softmax')(dropout2)

model = tf.keras.Model(inputs = inp, outputs = out)

sgd = tf.keras.optimizers.SGD(lr= .0001)
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam,
                loss='binary_crossentropy',
                metrics=['accuracy'])

# Summarize convolutional neural network architecture
model.summary()