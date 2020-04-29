# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:47:24 2020
Format data
@author: Aditya Ojha

Goal: take the images of cats and dogs
put cats in cats folder
put dog images in dogs folder
put the cats and dogs folder into a "CatDogTrain" folder

note 4/16/2020: it worked; data is now formatted
"""
import os
import shutil
#This function creates a directory if it doesn't exist, or lets you know if a directory exists

def createDir(directory):
    try:
        os.mkdir(directory)
        print("Directory "+directory+" Created")
    except FileExistsError:
        print("Directory "+directory+" already exists")
    

direct = "C:/Users/Monica/Desktop/KaggleCatsDogs"
createDir(direct)
    
train = direct + "/train"
CDTrain_dir = direct +"/CatDogTrain"
createDir(CDTrain_dir)

cats_dir = CDTrain_dir +"/cats"
dogs_dir = CDTrain_dir +"/dogs"
createDir(cats_dir)
createDir(dogs_dir)

train_dir = direct + "/train"

img_arr = os.listdir(train_dir)
for img in img_arr:
    img_path = os.path.join(train_dir,img)
    if("cat" in img_path):
        shutil.move(img_path,cats_dir)
    elif("dog" in img_path):
        shutil.move(img_path,dogs_dir)
    #break#want to test if it works ornot
