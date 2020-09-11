## Mods

import numpy as np

import os, keras, pickle

import tensorflow as tf

from keras.utils import np_utils, to_categorical

from keras.models import Sequential

from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout

from keras.optimizers import Adam

from keras.backend.tensorflow_backend import set_session

from keras.callbacks import TensorBoard

 

## Option of TF

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

set_session(sess)

 

## Import pictures

cloud_data = []

IMG_SIZE = 227

categories = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]

base_dir = "/home/kevin/Pro/tensorflow/CloudProject/src/"

 

## Load pickle

X = pickle.load(open(os.path.join(base_dir, "X_jeff_train_color.pkl"), 'rb'))

y = pickle.load(open(os.path.join(base_dir, "y_jeff_train_color.pkl"), 'rb'))

 

## CNN

X = X/255.0

y = to_categorical(y)

DROP = 0.2

BATCH = 32

EPOCH = 500

top_k = 1

 

def top_accuracy(y_true, y_pred):

    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=top_k)

 

def AlexNet(width, height, depth, classes):

   

    model = Sequential()

   

    #First Convolution and Pooling layer

    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(width,height,depth),padding='valid',activation='relu'))

    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

   

    #Second Convolution and Pooling layer

    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu'))

    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

   

    #Three Convolution layer and Pooling Layer

    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))

    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))

    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))

    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

   

    #Fully connection layer

    model.add(Flatten())

    model.add(Dense(4096,activation='relu'))

    model.add(Dropout(DROP))

    model.add(Dense(4096,activation='relu'))

    model.add(Dropout(DROP))

    model.add(Dense(1000,activation='relu'))

    model.add(Dropout(DROP))

   

    #Classfication layer

    model.add(Dense(classes,activation='softmax'))

 

    return model

 

AlexNet_model = AlexNet(IMG_SIZE, IMG_SIZE, 3, len(categories))

AlexNet_model.summary()

AlexNet_model.compile(

    loss="categorical_crossentropy",

    optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),

#    metrics=["accuracy", "top_k_categorical_accuracy"]

    metrics=["accuracy", top_accuracy]

    )

 

#import time

#NAME = "cloud_alex-{}".format(int(time.time()))

#tb = TensorBoard(log_dir="logs/{}".format(NAME))

#AlexNet_model.fit(X, y, batch_size=32, validation_split=0.2, epochs=100, callbacks=[tb])

 

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

img_gen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

    )

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, shuffle = True)

 

NAME = "cloud_alex_train_{}_{}_{}_top{}".format(BATCH, EPOCH, DROP, top_k)

tb = TensorBoard(log_dir=os.path.join(base_dir, "logs/{}".format(NAME)))

History = AlexNet_model.fit_generator(

    img_gen.flow(X_train*255, y_train, batch_size = BATCH),

    steps_per_epoch = len(X_train)//BATCH,

    validation_data = (X_test,y_test),

    epochs = EPOCH,

    callbacks=[tb])

 

print("Testing before saving:", AlexNet_model.predict(X_test[0:3]))

print("True answers:", y_test[0:3])

 

model_name = os.path.join(base_dir, "CloudAlex_Jeff_train_top{}.h5".format(top_k))

AlexNet_model.save(model_name)