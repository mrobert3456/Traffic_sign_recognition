# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:32:30 2021

@author: mrobe
"""

import numpy as np
import h5py
import pydot

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, AvgPool2D
from keras.utils import plot_model

from keras.utils.np_utils import to_categorical
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback


model = Sequential()

# Adding first pair
model.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPool2D())

# Adding second pair
model.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D())

# Adding third pair
model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D())

# Adding fourth pair
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D())

# Adding fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(43, activation='softmax')) # output

# Compiling created model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.save('ts' + '/' + 'model_ts_rgb.h5')

print('model is compiled successfully')

model.summary()

"""TRAINING PHASE"""

# Defining number of epochs
epochs = 50


# Defining schedule to update learning rate
learning_rate = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** (x + epochs), verbose=1)


# training dataset
datasets = ['dataset_ts_rgb_255_mean.hdf5',
            'dataset_ts_rgb_255_mean_std.hdf5']

# Defining list to collect results in
h = []

#training
for i in range(2):
   #open dataset
    with h5py.File('ts' + '/' + datasets[i], 'r') as f:
        # Extracting saved arrays for training by appropriate keys
        x_train = f['x_train']  # HDF5 dataset
        y_train = f['y_train']  # HDF5 dataset

        # Converting them into Numpy arrays
        x_train = np.array(x_train)  # Numpy arrays
        y_train = np.array(y_train)  # Numpy arrays

        # Extracting saved arrays for validation by appropriate keys
        x_validation = f['x_validation']  # HDF5 dataset
        y_validation = f['y_validation']  # HDF5 dataset
        # Converting them into Numpy arrays
        x_validation = np.array(x_validation)  # Numpy arrays
        y_validation = np.array(y_validation)  # Numpy arrays

    # Check point
    print('Following dataset is successfully opened:        ', datasets[i])

    # Preparing classes to be passed into the model
    # Transforming them from vectors to binary matrices
    # It is needed to set relationship between classes to be understood by the algorithm
    # Such format is commonly used in training and predicting
    y_train = to_categorical(y_train, num_classes=43)
    y_validation = to_categorical(y_validation, num_classes=43)

    # Check point
    print('Binary matrices are successfully created:        ', datasets[i])

    #saving weights
    best_weights_filepath = 'ts' + '/' + 'w_1' + datasets[i][7:-5] + '.h5'

    # Formatting options to save all weights for every epoch
    # 'ts' + '/' + 'w_1' + datasets[i][7:-5] + '_{epoch:02d}_{val_accuracy:.4f}' + '.h5'

    # Defining schedule to save best weights
    best_weights = ModelCheckpoint(filepath=best_weights_filepath,
                                   save_weights_only=True,
                                   monitor='val_accuracy',
                                   mode='max',
                                   save_best_only=True,
                                   period=1,
                                   verbose=1)

    # Check point
    print('Schedule to save best weights is created:        ', datasets[i])

    # Checking if RGB dataset is opened
    if i <= 1:
        # Training RGB model with current dataset
        temp = model.fit(x_train, y_train,
                                batch_size=50,
                                epochs=epochs,
                                validation_data=(x_validation, y_validation),
                                callbacks=[learning_rate, best_weights],
                                verbose=1)

        # Adding results of  model for current RGB dataset in the list
        h.append(temp)

        # Check points
        print('model for RGB is successfully trained on:    ', datasets[i])
        print('Trained weights for RGB are saved successfully:  ', 'w_1' + datasets[i][7:-5] + '.h5')
        print()

# Resulted accuracies of all Traffic Signs datasets
for i in range(2):
    print('T: {0:.5f},  V: {1:.5f},  D: {2}'.format(max(h[i].history['accuracy']),
                                                    max(h[i].history['val_accuracy']),
                                                    datasets[i][8:-5]))
