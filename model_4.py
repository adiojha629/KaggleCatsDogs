# -*- coding: utf-8 -*-
"""
Created on May 2 2020

@author: adioj
"""

#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Network from scratch

#Input layer
inp = keras.layers.Input(shape = (128,128,3))

#Block #1
b1_c1 = keras.layers.Conv2D(32,kernel_size = (3, 3), activation = 'relu', padding = 'same')(inp)
b1_c2 =keras.layers.Conv2D(32,kernel_size = (3, 3), activation = 'relu', padding = 'same')(b1_c1)
b1_pool = keras.layers.MaxPool2D(pool_size = (2,2))(b1_c2)

#Block #2
b2_c1 = keras.layers.Conv2D(64,kernel_size = (3, 3), activation = 'relu', padding = 'same')(b1_pool)
b2_c2 = keras.layers.Conv2D(64,kernel_size = (3, 3), activation = 'relu', padding = 'same')(b2_c1)
b2_pool = keras.layers.MaxPool2D(pool_size = (2,2))(b2_c2)

#Block #3
b3_c1 = keras.layers.Conv2D(128,kernel_size = (3, 3), activation = 'relu', padding = 'same')(b2_pool)
b3_c2 = keras.layers.Conv2D(128,kernel_size = (3, 3), activation = 'relu', padding = 'same')(b3_c1)
b3_c3 = keras.layers.Conv2D(128,kernel_size = (3, 3), activation = 'relu', padding = 'same')(b3_c2)

flat = keras.layers.Flatten()(b3_c3)

hidden1 = keras.layers.Dense(256, activation = 'relu')(flat)

dropout1 = keras.layers.Dropout(rate = 0.3)(hidden1)

hidden2 = keras.layers.Dense(256, activation = 'relu')(dropout1)

dropout2 = keras.layers.Dropout(rate = 0.3)(hidden2)

out = keras.layers.Dense(2, activation = 'softmax')(dropout2)



model = keras.Model(inputs = inp, outputs = out)

sgd = keras.optimizers.SGD(lr= .0001)
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam,
                loss='binary_crossentropy',
                metrics=['accuracy'])

# Summarize convolutional neural network architecture
model.summary()

""" Generate images"""

train_datagen = ImageDataGenerator(
        rescale = 1.0 / 255, # rescale the RGB values down to prevent overfitting
        rotation_range = 20, #Rotate some images to reduce overfitting
        width_shift_range = 0.1, #translate the image randomly by +/- 0.1 of the width of image
        height_shift_range = 0.1,
        shear_range = 0.2, #randomly distort the image by 0.2 (shear is a mathematical function)
        zoom_range = 0.2, #randomly zoom in on the image
        horizontal_flip = True,
        vertical_flip = True)
val_datagen = ImageDataGenerator (
        rescale = 1.0 / 255)

train_data_dir = "C:/Users/Default/Desktop/ML_data_sheets/model_testing"
img_height = 128
img_width = 128
val_data_dir = "C:/Users/Default/Desktop/ML_data_sheets/model_validation"

samples = 2000
epochs = 20
batch_size = 4

"""Generating training and validation images"""

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = 'categorical'
        )
val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = 'categorical')


"""Fit the imges to the training set"""

history = model.fit_generator(
        train_generator,
        steps_per_epoch = samples,
        epochs = epochs,
        validation_data = val_generator,
        validation_steps = samples
        )

