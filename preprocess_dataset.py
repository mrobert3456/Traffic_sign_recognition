# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 11:34:48 2021

@author: mrobe
"""

import pandas as pd
import numpy as np
import h5py
import cv2
import os

from sklearn.utils import shuffle
from tqdm import tqdm

"""Getting the ROI"""

# going through all files and directories in GTSRB dir
for curr_dir, dirs, files in os.walk('GTSRB'):
    for f in files:
        # need to obtain the excel file, which contains the informations about the images
        # ROI vertices, shape, name, etc
        if f.endswith('.csv'):
            csv_path = curr_dir + '/' + f

            dframe = pd.read_csv(csv_path, ';')
            drows = dframe.shape[0]  # how many images we have

            # ROI-d objects will be stored in this
            x_train = np.zeros((1, 48, 48, 3))  # image number, height, width, color channel

            # class numbers
            y_train = np.zeros(1)

            # store the currently ROI-d images, then it will be added vertically to the x, y_train
            x_temp = np.zeros((1, 48, 48, 3))
            # current class numbers
            y_temp = np.zeros(1)

            first_ts = True  # with that x,y_train will be updated once at the first ROI

            # iterating through the images and get ROI
            # tqdm will show a trackbar of the process
            for i in tqdm(range(drows)):

                img_path = curr_dir + '/' + dframe.loc[i, 'Filename']  # get the current image

                img = cv2.imread(img_path)  # read img

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                class_idx = dframe.loc[i, 'ClassId']  # get the class id

                # get ROI vertices
                x_left = dframe.loc[i, 'Roi.X1']
                y_left = dframe.loc[i, 'Roi.Y1']
                x_right = dframe.loc[i, 'Roi.X2']
                y_right = dframe.loc[i, 'Roi.Y2']

                # Getting ROI from imgs
                roi_img = img[y_left:y_right, x_left:x_right]

                roi_img = cv2.resize(roi_img, (48, 48), interpolation=cv2.INTER_CUBIC)  # Inter_cubic enlarges the img after resizing

                # if it is the first ROI-d img, then save it to the main arrays
                if first_ts:
                    x_train[0, :, :, :] = roi_img
                    y_train[0] = class_idx
                    first_ts = False
                else:
                    x_temp[0, :, :, :] = roi_img
                    y_temp[0] = class_idx

                    # adding current ROI-d img to the main array
                    x_train = np.concatenate((x_train, x_temp), axis=0)
                    y_train = np.concatenate((y_train, y_temp), axis=0)

            file_name = f[:-4]  # getting the save file names
            # saving ROI-d img-s to binary files
            # there will be 43 HDF5 files, each contains the corresponding traffic sign class's images
            with h5py.File('GTSRB/' + file_name + '.hdf5', 'w') as interm_f:
                interm_f.create_dataset('x_train', data=x_train, dtype='f')
                interm_f.create_dataset('y_train', data=y_train, dtype='i')

"""end of Getting the ROI"""
"""Getting the binary files and save it to one common array"""

# we need intermediate binary files to prevent memory overload

with h5py.File('GTSRB' + '/' + 'GT-final_test.hdf5', 'r') as f:
    x_train = f['x_train']  # HDF5 format
    y_train = f['y_train']

    # convert to np array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

# getting the rest 43 HDF5 files and save it to one HDF5 file
for curr_dir, dirs, files in os.walk('GTSRB'):
    for f in files:
        if f.endswith('.hdf5') and f != 'GT-final_test.hdf5':
            with h5py.File('GTSRB' + '/' + f, 'r') as intm_f:
                x_temp = intm_f['x_train']
                y_temp = intm_f['y_train']  # hdf5 format

                # convert to np array
                x_temp = np.array(x_temp)
                y_temp = np.array(y_temp)

                # adding to the main array
                x_train = np.concatenate((x_train, x_temp), axis=0)
                y_train = np.concatenate((y_train, y_temp), axis=0)

                print("Done: ", f)  # printing the processed filenames

""" end of Getting the binary files and save it to one common  numpy array"""

"""Suffle data along the x axis"""
x_train, y_train = shuffle(x_train, y_train)  # need to shuffle for training
# the cnn may find some connections of the orders of the incoming sings, and it reduces accuracy


"""Splitting dataset into train, validation and test"""

# first 30% for training
x_temp = x_train[:int(x_train.shape[0] * 0.3), :, :, :]
y_temp = y_train[:int(y_train.shape[0] * 0.3)]

# last 70% for training
x_train = x_train[int(x_train.shape[0] * 0.3):, :, :, :]
y_train = y_train[int(y_train.shape[0] * 0.3):]

# first 80% to validation during training
x_validation = x_temp[:int(x_temp.shape[0] * 0.8), :, :, :]
y_validation = y_temp[:int(y_temp.shape[0] * 0.8)]

# last 20% to test
x_test = x_temp[int(x_temp.shape[0] * 0.8):, :, :, :]
y_test = y_temp[int(y_temp.shape[0] * 0.8):]

"""End of Splitting dataset into train, validation and test"""

"""Save array to one binary file"""

with h5py.File('dataset_ts.hdf5', 'w') as f:
    f.create_dataset('x_train', data=x_train, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    f.create_dataset('x_validation', data=x_validation, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='f')

    f.create_dataset('x_test', data=x_test, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')

"""End of Save array to one binary file"""
