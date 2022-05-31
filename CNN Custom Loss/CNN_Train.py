
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K


## Define File Locations (Images, Spectra, and CNN Model Save)
img_path = '/.../Images_train/*.png'
spectra_path = '/.../cross_length_train.csv'
save_dir = '/.../model.h5'

## Load Images (CNN Input)
def loadImages(path):
    loadedImages = []
    filesname = glob.glob(path)
    filesname.sort()
    for imgdata in filesname:
        if os.path.isfile(os.path.splitext(os.path.join(path, imgdata))[0] + ".png"):
            img_array = cv2.imread(os.path.join(path, imgdata))
            img_array = np.float32(img_array)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)       
            loadedImages.append(gray)
    return loadedImages

imgs = loadImages(img_path)
CNN_input = np.array(imgs).reshape(len(imgs),128,128,1)
## Load Spectra from Excel (CNN Output)
CNN_output = np.array(np.float32(pd.read_csv(spectra_path, header = 0, index_col=0)))


def custom_mean_squared_error(i):
    def loss(y_true, y_pred):
        total_loss = K.mean(K.square(y_true - y_pred), axis=-1)
        return total_loss
    return loss

#%%
# Split Data into Train and Test Sets
CNN_input_train, CNN_input_test, CNN_output_train, CNN_output_test = train_test_split(CNN_input, CNN_output, test_size = 0.24, random_state = 42)
print('data size after spliting')
print('CNN_input_train size: {}'.format(np.shape(CNN_input_train)))
print('CNN_input_test size: {}'.format(np.shape(CNN_input_test)))
print('CNN_output_train size: {}'.format(np.shape(CNN_output_train)))
print('CNN_output_test size: {}'.format(np.shape(CNN_output_test)))

#%%
# Define CNN Architecture
#model = Sequential()
l0 = Input(shape=(CNN_input_train.shape[1:]))
l1 = Conv2D(32, (3,3), padding = 'same')(l0)
l2 = BatchNormalization()(l1)
l3 = LeakyReLU(0.2)(l2)
l4 = AveragePooling2D(pool_size = (2,2), strides = 2)(l3)
l5 = Conv2D(64, (3,3), padding = 'same')(l4)
l6 = BatchNormalization()(l5)
l7 = LeakyReLU(0.2)(l6)
l8 = AveragePooling2D(pool_size = (2,2), strides = 2)(l7)
l9 = Conv2D(128, (3,3), padding = 'same')(l8)
l10 = BatchNormalization()(l9)
l11 = LeakyReLU(0.2)(l10)
l12 = AveragePooling2D(pool_size = (2,2), strides = 2)(l11)
l13 = Conv2D(256, (3,3), padding = 'same')(l12)
l14 = BatchNormalization()(l13)
l15 = LeakyReLU(0.2)(l14)
l16 = AveragePooling2D(pool_size = (2,2), strides = 2)(l15)
l17 = Conv2D(512, (3,3), padding = 'same')(l16)
l18 = BatchNormalization()(l17)
l19 = LeakyReLU(0.2)(l18)
l20 = AveragePooling2D(pool_size = (2,2), strides = 2)(l19)
l21 = Flatten()(l20)
l22 = Dense(800)(l21)
model = Model(l0,l22)

cnnopt = Adam()
model.compile(loss=custom_mean_squared_error(l0), optimizer = cnnopt, metrics = ['accuracy'])
print(model.summary())        

# Train and Save CNN
epochs = 500
batch_size = 16
validation_data = (CNN_input_test, CNN_output_test)
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', restore_best_weights=True)
history = model.fit(CNN_input_train, CNN_output_train, batch_size = batch_size, epochs = epochs, validation_data = validation_data)
score = model.evaluate(CNN_input_test, CNN_output_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(model.summary())
model.save(save_dir)

# Plot Losses
plt.rcParams['font.size'] = '18'
fig, ax = plt.subplots()
ax.plot(history.history['loss'], color = 'b', label = 'Training Loss')
ax.plot(history.history['val_loss'], color = 'r', label = 'Validation Loss')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
plt.legend(loc = 'upper right')
plt.ylim([0, 0.1])
plt.show()
