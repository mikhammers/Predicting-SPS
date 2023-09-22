# Predicting-SPS
The data and code in this repository was used for analysing data for the manuscript "Phenotyping and Predicting Wheat Spike Characteristics Using image Analysis and Machine Learning". All code for machine learning algorithms and labels for images is presented below. Please reach out for access to images used with this data.

{\rtf1\ansi\ansicpg1252\cocoartf2636
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh10660\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Author: Mik Hammers\
Created: September 22, 2023\
Modified: September 22, 2023\
\
README\
\
\
Data in this repository is intended to be used to train machine learning models to count spikelet numbers in wheat. All models in this repository utilized a remote server that had 500 gb of RAM. Some files, such as the ResNet152V2 and EfficientNetV2L, will require servers with high computing power. The basic CNN and VGG16 models are able to be run on something else that has less computing power, such as Google Colab Pro+. Machine learning code and image labels are available in this repository. Images are too large to store on GitHub, please reach out to the author for acquisition of image files.\
\
CONTACT\
\
If you have any problems, questions, or ideas, please contact the owner of this repository, Mik Hammers.\
\
OPERATING INSTRUCTIONS\
\
For use, the label datasets (ML_test_labels.csv, ML_train_labels.csv, ML_val_labels.csv) as well as the machine learning model code of interest will need to be downloaded from the repository. You will also need to reach out to receive the test, train, and validation image sets from the author. \
\
From here, the code should run through uninterrupted, granted all files are in the same directory. The end of the code will need to be modified to ensure the proper results text file and CSV file are printed. These have been left blank in the code for you to fill out with the file names of your choosing. The code is commented to help identify these areas.\
\
FILES\
\
hammers_ML_test_lables.csv : spikelet labels for test set images.\
\
hammers_ML_train_labels.csv : spikelet labels for training set images.\
\
hammers_ML_val_labels.csv : spikelet labels for validation set images.\
\
hammers_cnn5j.py : code for simple convolutional neural network (CNN) model for counting wheat spikelets.\
\
hammers_vgg16.py : code for pre-trained VGG16 model for counting wheat spikelets.\
\
hammers_resnet152v2.py : code for pre-trained ResNet152V2 model for counting wheat spikelets.\
\
hammers_efficientnetV2L.py : code for pre-trained EfficientNetV2L model for counting wheat spikelets.}
