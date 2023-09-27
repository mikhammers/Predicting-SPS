# -*- coding: utf-8 -*-
"""
Author: Mik Hammers
Date: 07/26/2023
Date modified: 09/18/2023

EfficientNetV2L model code.
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
X_test = list()
X_val = list()

#training image dataset
for filename in sorted(listdir('train_images')):
  #loading images
  img = image.imread('train_images/' + filename)
  #storing images
  X_train.append(img)
X_train = np.array(X_train)

#test image dataset
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

#preprocessing and model info retrieved from: https://keras.io/api/applications/efficientnet_v2/#efficientnetv2l-function

#resnet image preprocessing
tf.keras.applications.efficientnet_v2.preprocess_input(X_train)
tf.keras.applications.efficientnet_v2.preprocess_input(X_test)
tf.keras.applications.efficientnet_v2.preprocess_input(X_val)


print(y_test)


#import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = tf.keras.applications.EfficientNetV2L(
      include_top=False,
      weights="imagenet",
      input_tensor=None,
      input_shape = (600, 140, 3),
      pooling=max,
      classifier_activation="softmax"
)

#displaying current architecture
model.summary()

#dense layers for model
from keras.models import Sequential

model2 = Sequential()
model2.add(model)
model2.add(layers.Flatten())
model2.add(layers.Dense(512, activation = 'linear'))
model2.add(layers.Dense(1))
model2.summary()



#compile and train 
model2.compile(optimizer = 'adam',
               loss = tf.keras.losses.MeanSquaredError(reduction = 'auto', name = "mean_squared_error"),
               metrics=[tf.keras.metrics.MeanAbsoluteError()])#,
               #experimental_run_tf_function = run_opts)

#early stopping
earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 20, restore_best_weights = True)




#model fit
history = model2.fit(X_train, y_train, epochs = 100, validation_data=(X_val, y_val), callbacks = [earlystop])

#estimate spikelet number on test image dataset
results = model2.predict(X_test)

#saving results in a CSV
np.savetxt("EfficientNetV2L.csv", results, delimiter= ",") #CHNAGE FILE NAME HERE

print(results)

#using saved model
test_loss, test_acc = model2.evaluate(X_test,  y_test, verbose=2)
print(f"test loss: {test_loss}")
print(f"test acc: {test_acc}")

#writing accuracy and loss results from forward validation in a text file
file = open("EfficientNetV2L_output.txt", "a") #CHANGE FILE NAME HERE
file.write(f"test loss: {test_loss}")
file.write(f"test acc: {test_acc}")

file.close()



