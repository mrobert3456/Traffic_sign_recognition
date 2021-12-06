# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 14:54:03 2021

@author: mrobe
"""

import numpy as np
import h5py


"""Preprocess TSD"""
with h5py.File('dataset_ts.hdf5','r') as f:

    #for training
    x_train = f['x_train']  # HDF5
    y_train = f['y_train']  # HDF5

    # Converting them into Numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #for validation
    x_validation = f['x_validation']  # HDF5
    y_validation = f['y_validation']  # HDF5

    # Converting them into Numpy arrays
    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)

    #for testing
    x_test = f['x_test']  # HDF5
    y_test = f['y_test']  # HDF5

    # Converting them into Numpy arrays
    x_test = np.array(x_test)
    y_test = np.array(y_test)

print('Numpy arrays of Traffic Signs Dataset')
print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)
print()

# apply normalization by divide every pixel by 255
x_train_norm=x_train/255.0
x_valid_norm=x_validation/255.0
x_test_norm=x_test/255.0


#saving it to new binary files
with h5py.File('ts'+'/'+'dataset_ts_rgb_255.hdf5','w') as f:
    # for training
    f.create_dataset('x_train', data=x_train_norm, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    #for validation
    f.create_dataset('x_validation', data=x_valid_norm, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    #  for testing
    f.create_dataset('x_test', data=x_test_norm, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')


# mean img calc from training ds
mean_rgb_ds_ts = np.mean(x_train_norm,axis=0)

# keep data centralized around 0, this will speed up training
x_train_norm_mean = x_train_norm-mean_rgb_ds_ts
x_valid_norm_mean = x_valid_norm-mean_rgb_ds_ts
x_test_norm_mean = x_test_norm-mean_rgb_ds_ts

#saving mean ds
with h5py.File('ts' + '/' + 'mean_rgb_dataset_ts.hdf5', 'w') as f:
    f.create_dataset('mean', data=mean_rgb_ds_ts, dtype='f')

#saving all to one binary file
with h5py.File('ts' + '/' + 'dataset_ts_rgb_255_mean.hdf5', 'w') as f:
    #  for training
    f.create_dataset('x_train', data=x_train_norm_mean, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    #  for validation
    f.create_dataset('x_validation', data=x_valid_norm_mean, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    #  for testing
    f.create_dataset('x_test', data=x_test_norm_mean, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')



#calculate standard deviation from training dataset
# the point is to scale pixel values to a smaller range to speed up training
std_rgb_ds_ts = np.std(x_train_norm_mean,axis=0)

x_train_norm_mean_std = x_train_norm_mean/std_rgb_ds_ts
x_valid_norm_mean_std = x_valid_norm_mean/std_rgb_ds_ts
x_test_norm_mean_std = x_test_norm_mean/std_rgb_ds_ts

#save std_ds
with h5py.File('ts' + '/' + 'std_rgb_dataset_ts.hdf5', 'w') as f:
    # Saving Numpy array for Mean Image
    f.create_dataset('std', data=std_rgb_ds_ts, dtype='f')


#saving all to one binary file
with h5py.File('ts' + '/' + 'dataset_ts_rgb_255_mean_std.hdf5', 'w') as f:
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train_norm_mean_std, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_valid_norm_mean_std, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test_norm_mean_std, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')

print('Original:            ', x_train_norm[0, 0, :5, 0])
print('- Mean Image:        ', x_train_norm_mean[0, 0, :5, 0])
print('/ Standard Deviation:', x_train_norm_mean_std[0, 0, :5, 0])
print()

# Check point
# Printing some values of Mean Image and Standard Deviation
print('Mean Image:          ', mean_rgb_ds_ts[0, :5, 0])
print('Standard Deviation:  ', std_rgb_ds_ts[0, :5, 0])
print()
