import numpy as np
import pandas as pd
import cv2
import os

number_epoch=100
step_per_epoch_training=20
step_per_epoch_validation=20

batch_size_training=15
batch_size_testing=2
batch_size_validation=2


import keras
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Sequential
from keras import optimizers
from keras import regularizers
from keras.layers.normalization import BatchNormalization 

def cnn(input_shape=(200,150,3)):
    model=Sequential()
    model.add(Conv2D(256,kernel_size=(4,4),activation="relu",input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,4)))
    model.add(Conv2D(128,(3,5),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(2,2),activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dropout(.5))
    model.add(Dense(64))
    model.add(Dense(2,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    model.summary()
    return model
    
model=cnn()
    
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

data_generator=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=data_generator.flow_from_directory("data/train",
                                                   target_size=(200,150),
                                                   batch_size=batch_size_training,
                                                   class_mode="categorical")
                                                   
validation_generator=data_generator.flow_from_directory("data/test",
                                                   target_size=(200,150),
                                                   batch_size=batch_size_validation,
                                                   class_mode="categorical")
history=model.fit_generator(train_generator,
                            steps_per_epoch=step_per_epoch_training,
                            epochs=number_epoch,
                            validation_data=validation_generator,
                            validation_steps=step_per_epoch_validation )
                            

