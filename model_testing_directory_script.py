# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:21:52 2020

@author: adioj
@Goal:
    create a "model_testing" directory which as two folders
        "cats_model_testing" which has 1000 randomly selected Cat images
        "dogs_model_testing" which has 1000 randomly selected Dog images

"""
""" Libraries Used """
import numpy as np
import random 
import os
import shutil

""" Functions Needed """
def createDir(directory):
    try:
        os.mkdir(directory)
        print("Directory "+directory+" Created")
    except FileExistsError:
        print("Directory "+directory+" already exists")

"""Script itself """
#Variables
direct = "C:/Users/Monica/Desktop/KaggleCatsDogs"
createDir(direct)
CDTrain_dir = direct +"/CatDogTrain"
createDir(CDTrain_dir)
cats_dir = CDTrain_dir +"/cats"
dogs_dir = CDTrain_dir +"/dogs"
createDir(cats_dir)
createDir(dogs_dir)

#Get the number of images in training data
num_cats = len(os.listdir(cats_dir))
num_dogs = len(os.listdir(dogs_dir))

#Get 1000 random indexs, not repeating
indexs_cats = random.sample(range(num_cats), 1000)
indexs_dogs = random.sample(range(num_dogs), 1000)

#Make the model_testing directory
m_ting_dir = direct + "/model_testing"
createDir(m_ting_dir)

#make the cats_model_testing folder
c_m_ting_dir = m_ting_dir + "/cats_model_testing"
createDir(c_m_ting_dir)
#make the dogs_model_testing folder
d_m_ting_dir = m_ting_dir + "/dogs_model_testing"
createDir(d_m_ting_dir)


#now add the images to these new directories
if(len(os.listdir(c_m_ting_dir)) == 0):#only add 1000 images if there are no images in there
    cats_entrys = os.listdir(cats_dir);
    for i in indexs_cats:
        img = cats_entrys[i]
        img_path = os.path.join(cats_dir,img)
        shutil.copy(img_path,c_m_ting_dir)

if(len(os.listdir(d_m_ting_dir)) == 0):#only add 1000 images if there are no images in there
    dogs_entrys = os.listdir(dogs_dir);
    for i in indexs_dogs:
        img = dogs_entrys[i]
        img_path = os.path.join(dogs_dir,img)
        shutil.copy(img_path,d_m_ting_dir)
    
""" Now generate validation images """
indexs_cat_val = random.sample(range(num_cats), 1000)
indexs_dog_val = random.sample(range(num_dogs), 1000)

#Make the model_validation directory
m_val_dir = direct + "/model_validation"
createDir(m_val_dir)

#make the cats_val_testing folder
c_m_val_dir = m_val_dir + "/cats_val"
createDir(c_m_val_dir)
#make the dogs_val_testing folder
d_m_val_dir = m_val_dir + "/dogs_val"
createDir(d_m_val_dir)

#now add the images to these new directories
if(len(os.listdir(c_m_val_dir)) == 0):#only add 1000 images if there are no images in there
    cats_entrys = os.listdir(cats_dir);
    for i in indexs_cat_val:
        img = cats_entrys[i]
        img_path = os.path.join(cats_dir,img)
        shutil.copy(img_path,c_m_val_dir)

if(len(os.listdir(d_m_val_dir)) == 0):#only add 1000 images if there are no images in there
    dogs_entrys = os.listdir(dogs_dir);
    for i in indexs_dog_val:
        img = dogs_entrys[i]
        img_path = os.path.join(dogs_dir,img)
        shutil.copy(img_path,d_m_val_dir)



