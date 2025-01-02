#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:44:57 2024
@author: chris k
"""

import pandas as pd
import numpy as np


"""
Training Data

Read in csv in corresponding folder with "Training" prefix

"""

TrainingDataPath='SA/'
TrainingData='Training-a1-a3.csv'
BatchSize=10
NumEpoch=10

dataset_train = pd.read_csv(TrainingDataPath+TrainingData, header=None)
X = dataset_train.iloc[:, 0:-2].values
label_column = dataset_train.iloc[:, -2].values
y = []
for i in range(len(label_column)):
    if label_column[i] == 'normal':
        y.append(0)
    else:
        y.append(1)

y_train = np.array(y)

#This sections gets rid of the categorical names and puts a vector insetad 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#columns 1,2,3 correspond to protocol, subtype, and flag 
ct = ColumnTransformer(
    #added ignore parameter because test set may have differing categorical values 
    [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), [1,2,3])],   
    remainder='passthrough'                        
)
x_train = np.array(ct.fit_transform(X), dtype=np.float)

#print(X)

"""

Testing Data

Read in csv in corresponding folder with "Testing" prefix
Same general pre-processing as 

"""

TestingDataPath='SA/'
TestingData='Testing-a2-a4.csv'

dataset_test = pd.read_csv(TestingDataPath+TestingData, header=None)
X2 = dataset_test.iloc[:, 0:-2].values
label_column2 = dataset_test.iloc[:, -2].values
y2 = []
for i in range(len(label_column2)):
    if label_column2[i] == 'normal':
        y2.append(0)
    else:
        y2.append(1)
        
y_test = np.array(y2)
x_test = np.array(ct.transform(X2), dtype=np.float)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)  # Scaling to the range [0,1]
X_test = sc.fit_transform(x_test)


from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer, 6 nodes, input_dim specifies the number of variables
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X_train[0])))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# sigmoid on the output layer is to ensure the network output is between 0 and 1
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN, 
# Gradient descent algorithm “adam“, Reference: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifierHistory = classifier.fit(X_train, y_train, batch_size = BatchSize, epochs = NumEpoch)
# evaluate the keras model for the provided model and dataset
loss, accuracy = classifier.evaluate(X_train, y_train)
print('Print the loss and the accuracy of the model on the dataset')
print('Loss [0,1]: %.4f' % (loss), 'Accuracy [0,1]: %.4f' % (accuracy))


"""
Making predictions and evaluating the model
"""


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.9)   # y_pred is 0 if less than 0.9 or equal to 0.9, y_pred is 1 if it is greater than 0.9
# summarize the first 5 cases
#for i in range(5):
#	print('%s => %d (expected %d)' % (X_test[i].tolist(), y_pred[i], y_test[i]))

# Making the Confusion Matrix
# [TN, FP ]
# [FN, TP ]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Print the Confusion Matrix:')
print('[ TN, FP ]')
print('[ FN, TP ]=')
print(cm)

"""
Data Visualization
"""

# Import matplot lib libraries for plotting the figures. 
import matplotlib.pyplot as plt

# Accuracy plot
print('Plot the accuracy')
# Keras 2.2.4 recognizes 'acc' and 2.3.1 recognizes 'accuracy'
# use the command python -c 'import keras; print(keras.__version__)' on MAC or Linux to check Keras' version
plt.plot(classifierHistory.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('accuracy_sample.png')
plt.show()

#Loss plot
print('Plot the loss')
plt.plot(classifierHistory.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('loss_sample.png')
plt.show()



#Addition analysis to answer question 2
#Attacks subclasses from dataExtractor.py
attacks_subClass = [['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm'], 
     ['ipsweep', 'mscan', 'portsweep', 'saint', 'satan','nmap'],
     ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'],
     ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']
     ]

# Convert y_pred to integer for easier comparison
y_pred_int = y_pred.astype(int)

# Identify indices for A2 attacks in the test set
A2_indices = [i for i, lbl in enumerate(label_column2) if str.lower(lbl) in attacks_subClass[1]]
# Identify indices for A4 attacks in the test set
A4_indices = [i for i, lbl in enumerate(label_column2) if str.lower(lbl) in attacks_subClass[3]]

# Compute accuracy for A2 attacks
if len(A2_indices) > 0:
    A2_correct = np.sum(y_pred_int[A2_indices] == y_test[A2_indices])
    A2_accuracy = A2_correct / len(A2_indices)
    print("A2 Attack Accuracy: {:.4f}".format(A2_accuracy))
else:
    print("No A2 attacks found in the test set.")

# Compute accuracy for A4 attacks
if len(A4_indices) > 0:
    A4_correct = np.sum(y_pred_int[A4_indices] == y_test[A4_indices])
    A4_accuracy = A4_correct / len(A4_indices)
    print("A4 Attack Accuracy: {:.4f}".format(A4_accuracy))
else:
    print("No A4 attacks found in the test set.")