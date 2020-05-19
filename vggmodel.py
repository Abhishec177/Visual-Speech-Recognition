import keras,os
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

tr_data = ImageDataGenerator()
train_data = tr_data.flow_from_directory(directory='../data/',target_size=(175,175))

def predictor_model():
    model = Sequential()
    
    vggmodel = VGG16(weights='imagenet', include_top=True)
    #VGG16 Model implementation    
    model.add(vggmodel)
    #classifier model to put on top of the convolutional model
    classifier_model = Sequential()
    classifier_model.add(Flatten(input_shape=model.output_shape[1:]))
    classifier_model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.1)))
    classifier_model.add(Dropout(0.5))
    classifier_model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.1)))
    classifier_model.add(Dropout(0.5))
    classifier_model.add(Dense(20, activation='softmax')) 
    model.add(classifier_model)
    return model

    #model = predictor_model()
    #model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    #model.summary()