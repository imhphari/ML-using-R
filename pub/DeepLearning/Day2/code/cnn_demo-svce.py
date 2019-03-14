from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as K

import sklearn
from keras.models import load_model
import pandas as pd  
from keras.preprocessing import image
from PIL import Image
import os
import keras
import keras.utils
from keras import utils as np_utils

#from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


import csv






# dimensions of our images.
img_width, img_height = 76, 76


#########################
### Tunables
#########################
train_data_dir = '/home/user/Nov13/data/train2'
test_data_dir = '/home/user/Nov13/data/test2'
nb_train_samples = 2410
nb_test_samples = 450
epochs = 5
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#########################
### Setup the model
#########################
# w2=(w1-F+2P)/(S)+1=>CONVWIDTH

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


#########################
### Setup the generators
###fits the model on batches with real-time data augmentation
#########################


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


#########################
### Build the model
#########################

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size)

#score = model.evaluate_generator(validation_generator, nb_val_samples/batch_size, workers=12)
#########################
### Performance evaluation
#########################
score = model.evaluate_generator(test_generator,nb_test_samples/batch_size)
print(" Total: ", len(test_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])


#########################
### Saving model for future use
#########################
model.save('firstmodel.h5')


#########################
### Predicting the classes
#########################

f1 = open("results.csv",'w')

for root, dirs, files in os.walk("/home/user/Nov13/data/test2", topdown=False):
    if root == "/home/user/Nov13/data/test2":
        for name in dirs:
            TEST_DIR="/home/user/Nov13/data/test2/"+name+"/"  
            print(TEST_DIR)
            img_file=os.listdir(TEST_DIR)
            for f in (img_file):
                img = Image.open(TEST_DIR+f)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                preds = model.predict_classes(x)
                print(preds[0])
                ptemp=str(preds[0]).replace("[", "")
                ptemp1=ptemp.replace("]", "")
                f1.write(name+"\t"+ ptemp1 +"\n")

f1.close()

