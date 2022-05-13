# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:18:22 2019

@author: cyyeu
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model

#load model
model = load_model('model.h5')

#Index data and generate prediction
i = 10
x = np.arange(0.5, 10.1, 0.1).reshape(1,96)
outputANN = model.predict(input_test[i].reshape(1,4))[0][:]
outputExcelTest = output_test[i].reshape(1,96)[0][:]

#Plot outputANN (prediction) and outputExcelTest (validation data)
fig, ax = plt.subplots()
ax.set(xlabel = 'Wavelength (μm)', ylabel = 'Absorption')
ax.plot(x[0], outputExcelTest, label = 'ground truth', linewidth = 4.0)
ax.plot(x[0], outputANN, label = 'predictions', linewidth = 4.0)
plt.xlim(0.5,10)
plt.ylim(0,1)
plt.legend()
plt.show()
