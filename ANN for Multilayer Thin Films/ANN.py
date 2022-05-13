# first neural network with keras tutorial
from keras.models import Sequential
from keras.layers import Dense, ReLU
from keras.callbacks.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# load the dataset
inputExcel = np.array(np.float32(pd.read_csv('C:/Users/cyyeu/Documents/Python/ANN/Fresnel/FresnelInput.csv')))
outputExcel = np.array(np.float32(pd.read_csv('C:/Users/cyyeu/Documents/Python/ANN/Fresnel/FresnelOutput.csv')))

# split into training and test data
input_train, input_test, output_train, output_test = train_test_split(inputExcel, outputExcel, test_size = 0.1, random_state = 42)
print('input_train size: {}'.format(np.shape(input_train)))
print('input_test size: {}'.format(np.shape(input_test)))
print('output_train size: {}'.format(np.shape(output_train)))
print('output_test size: {}'.format(np.shape(output_test)))

# define the keras model
model = Sequential()
model.add(Dense(20, input_dim=4))
model.add(ReLU())
model.add(Dense(20))
model.add(ReLU())
model.add(Dense(96))

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=0, mode='auto', restore_best_weights=True)
# history = model.fit(input_train, output_train, validation_data = (input_test, output_test), epochs = 500, verbose = 2, batch_size = 10)
history = model.fit(input_train, output_train, validation_data = (input_test, output_test), epochs = 1000, verbose = 2, callbacks = [es], batch_size = 10)

# evaluate the keras model
score = model.evaluate(input_test, output_test)
print('val_loss:', score[0])
print('val_accuracy:', score[1])

# Plot Losses
fig, ax1 = plt.subplots()
ax1.plot(history.history['loss'], color = 'b', label = 'Training Loss')
ax1.plot(history.history['val_loss'], color = 'r', label = 'Validation Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epochs')
plt.ylim(0,0.01)
plt.legend(loc = 'upper right')
plt.show()


model.save('model.h5')
model.summary()