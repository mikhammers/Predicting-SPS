# -*- coding: utf-8 -*-
"""
Author: Mik Hammers
Date created: 07/2023
Date modified: 08/12/2023


Basic regression convolutional neural network (CNN) with five alternating convolutional and pooling layers.
Access to label files can be found on GitHub repository. For access to images, please reach out to hammers@colostate.edu

"""

# Importing packages
from sklearn import datasets
import pandas as pd
import numpy as np
from numpy import asarray
import os
from os import listdir
import tensorflow as tf
from tensorflow import keras
from  matplotlib import pyplot as plt
from matplotlib import image
import matplotlib.image as mpimg
import os, io
from PIL import Image
from os import listdir
from matplotlib import image
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#Loading labels for the training, test, and validation datasets
y_train = pd.read_csv("ML_train_labels.csv", usecols = ["label"])
y_test = pd.read_csv("ML_test_labels.csv", usecols = ["label"])
y_val = pd.read_csv("ML_val_labels.csv", usecols = ["label"])

#Creating lists to store training, test, and validation image datasets
X_train = list()
X_val = list()
X_test = list()

#training image dataset
for filename in sorted(listdir('train_images')):
  #loading images
  img = image.imread('train_images/' + filename)
  #storing images
  X_train.append(img)
X_train = np.array(X_train)

#testing image dataset
for filename in sorted(listdir('test_images')):
  #loading images
  img2 = image.imread('test_images/' + filename)
  #storing images
  X_test.append(img2)
X_test = np.array(X_test)

#validation image dataset
for filename in sorted(listdir('val_images')):
  #loading images
  img3 = image.imread('val_images/' + filename)
  #storing images
  X_val.append(img3)
X_val = np.array(X_val)

print(y_test)


#normalizing image values to be between 0 and 1 for basic model
X_test = X_test/255.0
X_train = X_train/255.0
X_val = X_val/255.0

#import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

#basic 5 layer
model = models.Sequential()
model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(600, 140, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

#Regression dense layers for basic models
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='linear'))
model.add(layers.Dense(1, use_bias = True))
model.summary()

#compile and train for basic regression model
model.compile(optimizer='adam',
              loss = tf.keras.losses.MeanSquaredError(reduction = 'auto', name = "mean_squared_error"),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

#early stopping
earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights= True)

#basic model fit
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks = [earlystop])


#Make spikelet predictions for test set images
results = model.predict(X_test)
print(results)

#Save spikelet estimation results in a CSV file
np.savetxt("CNN5.csv", results, delimiter= ",") #EDIT YOUR FILE NAME HERE

#Evaluating forward validation accuracy and fit for model
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(f"test loss: {test_loss}")
print(f"test acc: {test_acc}")

#printing results into a text file
file = open("cnn5_output.txt", "a") #EDIT YOUR FILE NAME HERE
file.write(f"test loss: {test_loss}")
file.write(f"test acc: {test_acc}")

file.close()



