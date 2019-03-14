#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:32:19 2017

@author: mirnalini

"""

#conda config --add channels conda-forge
#conda install mahotas
#conda install -c conda-forge mahotas 
#Mahotas is a library of fast computer vision algorithms (all implemented in C++) operating over numpy arrays.


import cv2
import numpy as np
import os
import glob
import mahotas as mt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#from sklearn.svm import LinearSVC


train_path = "/home/user/Nov13/dataset/train_mel"
train_names = os.listdir(train_path)



train_path_nev= "/home/user/Nov13/dataset/train_naevs"
train_nav_names = os.listdir(train_path_nev)


# empty list to hold feature vectors and train labels
train_features = []
train_labels = []
test_features=[]
test_labels=[]
total_fea=[]
fixed_size = tuple((750, 750))

 # calculate haralick texture features for 4 types of adjacency
def extract_features(image):
       
        textures = mt.features.haralick(image)
        ht_mean = textures.mean(axis=0)
        return ht_mean

#Extracting moments of the images
def hu_moments(image):
    m=cv2.HuMoments(cv2.moments(gray))
    hu1=-np.sign(m)*np.log10(np.abs(m))
    hu=hu1.flatten()
    return(hu)

    

for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label =[1]
        
        image = cv2.imread(cur_path)
        plt.imshow(image)
               # convert the image to grayscale
        image = cv2.resize(image, fixed_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               # extract haralick texture from the image
        features = extract_features(gray)
        hu_mom=hu_moments(gray)
        #color_fea=fd_histogram(image)
        s=np.concatenate((features, hu_mom), axis=0)
       
       
       
        train_features.append(s)
        train_labels.append(cur_label)
#        



for train_nav_name in train_nav_names:
        cur_path = train_path + "/" + train_name
        cur_label =[0]
        # read the training image
        image = cv2.imread(cur_path)
        plt.imshow(image)
        image = cv2.resize(image, fixed_size)
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # extract haralick texture from the image
        features = extract_features(gray)
        hu_mom=hu_moments(gray)
        s=np.concatenate((features, hu_mom), axis=0)
        train_features.append(s)
        train_labels.append(cur_label)
        X_train=np.array(train_features)
        y_train=np.array(train_labels)








## loop over the test images
test_path = "/home/user/Nov13/dataset/test_mel"
for file in glob.glob(test_path + "/*.jpg"):
        # read the input image
        image = cv2.imread(file)
        cur_label = [1]
        # convert to grayscale
        image = cv2.resize(image, fixed_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = extract_features(gray)
        hu_mom=hu_moments(gray)
        s=np.concatenate((features, hu_mom), axis=0)
        test_features.append(s)
        test_labels.append(cur_label)

test_path1="/home/user/Nov13/dataset/test_naevs"
for file in glob.glob(test_path1 + "/*.jpg"):
        # read the input image
        image = cv2.imread(file)
        cur_label = [0]
        # convert to grayscale
        image = cv2.resize(image, fixed_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # extract haralick texture from the image
        features = extract_features(gray)
        hu_mom=hu_moments(gray)
        s=np.concatenate((features, hu_mom), axis=0)
        test_features.append(s)
        test_labels.append(cur_label)
       
        X_test= np.array(test_features)
        y_test=np.array(test_labels)
       
#normalization
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_test_maxabs=max_abs_scaler.fit_transform(X_test)



#floating point array
X_train = np.array(X_train_maxabs).astype(np.float32)
X_test  = np.array(X_test_maxabs).astype(np.float32)
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)



print("X_train",X_train.shape)
print("y_train", y_train.shape)
print(X_test.shape, y_test.shape)
#print("normalized",X_train_maxabs)


        
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import numpy as np

def simple_nn():

       
    model = Sequential()
        
    model.add(Dense(32, input_dim=20,kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16,kernel_initializer='random_normal', activation='relu'))
    model.add(Dense(16,kernel_initializer='random_normal', activation='relu'))
    model.add(Dense(8,kernel_initializer='random_normal', activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='random_normal',activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=50, verbose=2)
    return model

def crossvalidation():
    # define 10-fold cross validation test harness
    seed = 5
    model = KerasClassifier(build_fn=simple_nn, epochs=150, batch_size=10, verbose=2)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(model, X_train, y_train, cv=kfold)
    print(results)
    print(results.mean())     
        
        
def performance():
    
    model=simple_nn()
    score = model.evaluate(X_test, y_test, verbose=2)
    print("Accuracy:",score[1])
    preds = model.predict_classes(X_test)
    print("Predicted_probability",preds)
# 

performance()
#crossvalidation()

