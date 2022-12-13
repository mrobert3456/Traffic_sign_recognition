# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import h5py
import cv2
import os
from sklearn.utils import shuffle
from tqdm import tqdm
from os.path import exists

"""
    Generate augmented dataset and original dataset
    the results goes to either to the 'ts/aug' folder or to the 'ts/orig'
"""

def changeBrightness(img):
    """
    Change brightness in the input image
    """
    #  RGB to HSV colour space
    image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    n = np.random.choice([-1, 1])

    if n == -1:
        # darken image
        random_brightness = n * np.random.randint(low=2, high=6)
    elif n == 1:
        # brighten image
        random_brightness = np.random.randint(low=50, high=75)

    # adjust HSV image
    image_hsv[:, :, 2] += random_brightness

    # if value is above 255 or below 0 then use these boundry numbers
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2], 0, 255)

    # HSV image back to RGB colour space
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return image_rgb


def changeRotation(img):
    """
    Rotates the input image
    """
    # random angle for rotation
    angle = np.random.randint(low=5, high=15) * np.random.choice([-1, 1])

    height, width, channels = img.shape

    # centre point of input image
    centre_point = (int(width / 2), int(height / 2))

    # Calculating rotation Matrix
    affine_matrix = cv2.getRotationMatrix2D(centre_point, angle, scale=1)

    # Warping original image with rotation Matrix
    rotated_image = cv2.warpAffine(img, affine_matrix, (height, width))

    return rotated_image


def perspectiveChangeAlongX(img):
    """
    x axis perspective transform
    """
    # Getting shape of input image
    height, width, channels = img.shape

    # vertices of input image
    x_min = 0
    y_min = 0
    x_max = width
    y_max = height

    # str matrix
    src = np.float32([[x_min, y_min],  # top-left
                      [x_max, y_min],  # top-right
                      [x_min, y_max],  # bottom-left
                      [x_max, y_max]])  # bottom-right

    # dst matrix
    dst = np.float32([[x_min + 5, y_min + 5],  # top-left
                      [x_max - 5, y_min + 5],  # top-right
                      [x_min, y_max],  # bottom-left
                      [x_max, y_max]])  # bottom-right

    # perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Applying perspective transformation
    projected_image = cv2.warpPerspective(img, matrix, (height, width))

    return projected_image


def perspectiveChangeAlongY(img):
    """
       y axis perspective transformation
    """
    # Getting shape of input image
    height, width, channels = img.shape

    # vertices of input image
    x_min = 0
    y_min = 0
    x_max = width
    y_max = height

    # src matrix
    src = np.float32([[x_min, y_min],  # top-left
                      [x_max, y_min],  # top-right
                      [x_min, y_max],  # bottom-left
                      [x_max, y_max]])  # bottom-right

    # dst matrix
    dst = np.float32([[x_min, y_min],  # top-left
                      [x_max - 5, y_min + 5],  # top-right
                      [x_min, y_max],  # bottom-left
                      [x_max - 5, y_max - 5]])  # bottom-right

    # perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Applying perspective transformation
    projected_image = cv2.warpPerspective(img, matrix, (height, width))

    return projected_image


def createValidationAndTest(x_train, y_train, dbname):
    """
    70% goes to training -> 903*43 = 38829
    This method takes 903 img from every class, then assign 80% of (38829) it to validation and 20% of (38829) to test
    """
    x_temp = []
    y_temp = []
    db = 0
    indexes = []
    for i in tqdm(range(43)):
        db = 0
        for j in range(len(x_train)):
            if y_train[j] == i and db < 903:
                indexes.append(j)
                x_temp.append(x_train[j])
                y_temp.append(i)
                db = db + 1

    x_train = np.delete(x_train, indexes, axis=0)
    y_train = np.delete(y_train, indexes, axis=0)

    x_temp = np.array(x_temp)
    y_temp = np.array(y_temp)
    x_temp, y_temp = shuffle(x_temp, y_temp)
    print(x_train.shape)
    # Slicing first 80% of elements from temp Numpy arrays
    x_validation = x_temp[:int(x_temp.shape[0] * 0.8), :, :, :]
    y_validation = y_temp[:int(y_temp.shape[0] * 0.8)]

    # Slicing last 20% of elements from temp Numpy arrays
    x_test = x_temp[int(x_temp.shape[0] * 0.8):, :, :, :]
    y_test = y_temp[int(y_temp.shape[0] * 0.8):]

    with h5py.File('ts/' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('x_train', data=x_train, dtype='f')
        f.create_dataset('y_train', data=y_train, dtype='i')

        f.create_dataset('x_validation', data=x_validation, dtype='f')
        f.create_dataset('y_validation', data=y_validation, dtype='i')

        f.create_dataset('x_test', data=x_test, dtype='f')
        f.create_dataset('y_test', data=y_test, dtype='i')


def equalizeDatasetByAugmentation(dbname):
    with h5py.File('ts/aug/dataset_ts_merged.hdf5', 'r') as f:

        x_train = f['x_train']
        y_train = f['y_train']

        x_train = np.array(x_train)
        y_train = np.array(y_train)

    classesIndexes, classesFrequency = np.unique(y_train, return_counts=True)

    for i in range(len(classesIndexes)):

        # number of images needed to be generated
        number_of_images_to_add = np.max(classesFrequency) + 10 - classesFrequency[i]
        x_temp = []
        y_temp = []

        # Augmenting current class
        for j in tqdm(range(number_of_images_to_add)):
            # y array indexes for current class
            image_indexes = np.where(y_train == i)

            # Extracting only array itself
            image_indexes = image_indexes[0]

            # Get random image index
            n = np.random.randint(low=0, high=classesFrequency[i])

            # Getting random image from current class
            random_image = np.copy(x_train[image_indexes[n]])

            # random brightness changing
            random_image = changeBrightness(random_image)

            # Choosing transformation technique
            m = np.random.choice([1, 2, 3])

            # Applying rotation around centre point
            if m == 1:
                random_image = changeRotation(random_image)

            # Applying perspective x axis
            elif m == 2:
                random_image = perspectiveChangeAlongX(random_image)

            # Applying perspective y axis
            elif m == 3:
                random_image = perspectiveChangeAlongY(random_image)

            # Appending transformed image into the list
            x_temp.append(random_image)
            y_temp.append(i)

        x_temp = np.array(x_temp)
        y_temp = np.array(y_temp)

        # Concatenating to main arrays
        x_train = np.concatenate((x_train, x_temp), axis=0)
        y_train = np.concatenate((y_train, y_temp), axis=0)

    createValidationAndTest(x_train, y_train, dbname)


def preproccessTrainingDataset():
    """
    Getting ROI from images, and store them
    """
    for curr_dir, dirs, files in os.walk('GTSRB\\Final_Training\\Images'):
        for f in files:
            # obtain the excel file, which contains  informations about the images
            if f.endswith('.csv'):
                csv_path = curr_dir + '/' + f

                dframe = pd.read_csv(csv_path, ';')
                drows = dframe.shape[0]  # image number

                x_train = np.zeros((1, 48, 48, 3))  # image number, height, width, color channel
                y_train = np.zeros(1)

                # store the currently ROI-d images
                x_temp = np.zeros((1, 48, 48, 3))
                y_temp = np.zeros(1)

                first_ts = True  # with that x,y_train will be updated once at the first ROI
                # iterating through the images and get ROI
                print(curr_dir)
                for i in tqdm(range(drows)):

                    img_path = curr_dir + '/' + dframe.loc[i, 'Filename']  # get the current image

                    if exists(img_path):
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        class_idx = dframe.loc[i, 'ClassId']  # get the class id

                        # get ROI vertices
                        x_left = dframe.loc[i, 'Roi.X1']
                        y_left = dframe.loc[i, 'Roi.Y1']
                        x_right = dframe.loc[i, 'Roi.X2']
                        y_right = dframe.loc[i, 'Roi.Y2']

                        # Getting ROI from imgs
                        roi_img = img[y_left:y_right, x_left:x_right]

                        # resize original image
                        roi_img = cv2.resize(roi_img, (48, 48),
                                             interpolation=cv2.INTER_CUBIC)  # Inter_cubic enlarges the img after resizing

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

                with h5py.File('GTSRB/' + file_name + '.hdf5', 'w') as interm_f:
                    interm_f.create_dataset('x_train', data=x_train, dtype='f')
                    interm_f.create_dataset('y_train', data=y_train, dtype='i')
                print(file_name + " is processed")


def preprocessTestDataset():
    """
    Getting ROI from test images, and store them
    """
    # LOADING TEST IMAGES
    test_dir = 'GTSRB\\Final_Test\\Images'
    test_cvs = test_dir + '\\GT-final_test.csv'
    dframe = pd.read_csv(test_cvs, ';')
    drows = dframe.shape[0]  # how many images we have

    # ROI-d objects will be stored in this
    x_trainTest = np.zeros((1, 48, 48, 3))  # image number, height, width, color channel
    y_trainTest = np.zeros(1)

    # store the currently ROI-d images
    x_tempTest = np.zeros((1, 48, 48, 3))
    y_tempTest = np.zeros(1)

    first_ts = True  # with that x,y_train will be updated once at the first ROI

    # iterating through the images and get ROI
    for i in tqdm(range(drows)):
        img_path = test_dir + '/' + dframe.loc[i, 'Filename']  # get the current image

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        class_idx = dframe.loc[i, 'ClassId']  # get the class id

        # get ROI vertices
        x_left = dframe.loc[i, 'Roi.X1']
        y_left = dframe.loc[i, 'Roi.Y1']
        x_right = dframe.loc[i, 'Roi.X2']
        y_right = dframe.loc[i, 'Roi.Y2']

        # Getting ROI from imgs
        roi_img = img[y_left:y_right, x_left:x_right]

        roi_img = cv2.resize(roi_img, (48, 48),
                             interpolation=cv2.INTER_CUBIC)  # Inter_cubic enlarges the img after resizing

        # if it is the first ROI-d img, then save it to the main arrays
        if first_ts:
            x_trainTest[0, :, :, :] = roi_img
            y_trainTest[0] = class_idx
            first_ts = False
        else:
            x_tempTest[0, :, :, :] = roi_img
            y_tempTest[0] = class_idx

            # adding current ROI-d img to the main array
            x_trainTest = np.concatenate((x_trainTest, x_tempTest), axis=0)
            y_trainTest = np.concatenate((y_trainTest, y_tempTest), axis=0)

    with h5py.File('GTSRB/GT-test_final.hdf5', 'w') as interm_f:
        interm_f.create_dataset('x_trainTest', data=x_trainTest, dtype='f')
        interm_f.create_dataset('y_trainTest', data=y_trainTest, dtype='i')


def mergeDatasets():
    """
    Merge Test and Train dataset into one hdf5 file
    :return:
    """
    x_trainTest = np.zeros((1, 48, 48, 3))
    y_trainTest = np.zeros(1)  # np.zeros(1)

    x_train = np.zeros((1, 48, 48, 3))
    y_train = np.zeros(1)  # np.zeros(1)

    x_temp = np.zeros((1, 48, 48, 3))
    y_temp = np.zeros(1)

    with h5py.File('GTSRB/GT-test_final.hdf5', 'r') as f:
        x_trainTest = f['x_trainTest']
        y_trainTest = f['y_trainTest']

        x_trainTest = np.array(x_trainTest)
        y_trainTest = np.array(y_trainTest)

    # getting the rest 43 HDF5 files and save it to one HDF5 file
    for curr_dir, dirs, files in os.walk('GTSRB'):
        for f in files:
            if f.endswith('.hdf5') and f != 'GT-test_final.hdf5':
                with h5py.File('GTSRB/' + f, 'r') as intm_f:
                    x_temp = intm_f['x_train']
                    y_temp = intm_f['y_train']

                    x_temp = np.array(x_temp)
                    y_temp = np.array(y_temp)
                    print("Done: ", f)

                    x_train = np.concatenate((x_train, x_temp), axis=0)
                    y_train = np.concatenate((y_train, y_temp), axis=0)

    # add test datas to train datas
    x_train = np.concatenate((x_train, x_trainTest), axis=0)
    y_train = np.concatenate((y_train, y_trainTest), axis=0)

    # Save array to one binary file
    with h5py.File('ts/aug/dataset_ts_merged.hdf5', 'w') as f:
        f.create_dataset('x_train', data=x_train, dtype='f')
        f.create_dataset('y_train', data=y_train, dtype='i')


def mergeDatasetsWithTestDataset(dbname):
    """
    Merge Test and Train dataset into one hdf5 file and creates sub dataset for test, training and validation
    70% goes to training -> 51.840*07 =36288
    Remaining 30% to test and validation: 80% for validation and 20% for test
    """
    x_trainTest = np.zeros((1, 48, 48, 3))
    y_trainTest = np.zeros(1)

    x_validation = np.zeros((1, 48, 48, 3))
    y_validation = np.zeros(1)

    x_train = np.zeros((1, 48, 48, 3))
    y_train = np.zeros(1)

    x_temp = np.zeros((1, 48, 48, 3))
    y_temp = np.zeros(1)

    with h5py.File('GTSRB/GT-test_final.hdf5', 'r') as f:
        x_trainTest = f['x_trainTest']
        y_trainTest = f['y_trainTest']

        x_trainTest = np.array(x_trainTest)
        y_trainTest = np.array(y_trainTest)

    # getting the rest 43 HDF5 files and save it to one HDF5 file
    for curr_dir, dirs, files in os.walk('GTSRB'):
        for f in files:
            if f.endswith('.hdf5') and f != 'GT-test_final.hdf5':
                with h5py.File('GTSRB/' + f, 'r') as intm_f:
                    x_temp = intm_f['x_train']
                    y_temp = intm_f['y_train']

                    x_temp = np.array(x_temp)
                    y_temp = np.array(y_temp)
                    print("Done: ", f)
                    x_train = np.concatenate((x_train, x_temp), axis=0)
                    y_train = np.concatenate((y_train, y_temp), axis=0)

    # add test datas to train datas
    x_train = np.concatenate((x_train, x_trainTest), axis=0)
    y_train = np.concatenate((y_train, y_trainTest), axis=0)

    x_train, y_train = shuffle(x_train, y_train)
    print(y_train.shape)

    # Slicing first 30% of elements for validation and test
    x_temp = x_train[:int(x_train.shape[0] * 0.3), :, :, :]
    y_temp = y_train[:int(y_train.shape[0] * 0.3)]

    # Slicing last 70% of x_train and y_train
    x_train = x_train[int(x_train.shape[0] * 0.3):, :, :, :]
    y_train = y_train[int(y_train.shape[0] * 0.3):]

    # Slicing first 80% of x_temp and y_temp
    x_validation = x_temp[:int(x_temp.shape[0] * 0.8), :, :, :]
    y_validation = y_temp[:int(y_temp.shape[0] * 0.8)]

    # Slicing last 20% of x_temp and y_temp
    x_test = x_temp[int(x_temp.shape[0] * 0.8):, :, :, :]
    y_test = y_temp[int(y_temp.shape[0] * 0.8):]

    with h5py.File('ts/' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('x_train', data=x_train, dtype='f')
        f.create_dataset('y_train', data=y_train, dtype='i')

        f.create_dataset('x_validation', data=x_validation, dtype='f')
        f.create_dataset('y_validation', data=y_validation, dtype='f')

        f.create_dataset('x_test', data=x_test, dtype='f')
        f.create_dataset('y_test', data=y_test, dtype='f')


def preprocessRGBDataset(folder, dbname):
    """
    Normalization, mean subraction, std for gray images
    """
    with h5py.File('ts/' + folder + '/' + dbname + '.hdf5', 'r') as f:
        x_train = f['x_train']
        y_train = f['y_train']
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_validation = f['x_validation']
        y_validation = f['y_validation']
        x_validation = np.array(x_validation)
        y_validation = np.array(y_validation)

        x_test = f['x_test']
        y_test = f['y_test']
        x_test = np.array(x_test)
        y_test = np.array(y_test)

    # apply normalization by divide every pixel by 255
    x_train_norm = x_train / 255.0
    x_valid_norm = x_validation / 255.0
    x_test_norm = x_test / 255.0

    with h5py.File('ts/' + folder + '/norm_rgb_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('x_train', data=x_train_norm, dtype='f')
        f.create_dataset('y_train', data=y_train, dtype='i')

        f.create_dataset('x_validation', data=x_valid_norm, dtype='f')
        f.create_dataset('y_validation', data=y_validation, dtype='i')

        f.create_dataset('x_test', data=x_test_norm, dtype='f')
        f.create_dataset('y_test', data=y_test, dtype='i')

    # mean img calc from training ds
    mean_rgb_ds_ts = np.mean(x_train_norm, axis=0)

    # keep data centralized around 0, this will speed up training
    x_train_norm_mean = x_train_norm - mean_rgb_ds_ts
    x_valid_norm_mean = x_valid_norm - mean_rgb_ds_ts
    x_test_norm_mean = x_test_norm - mean_rgb_ds_ts

    # saving mean ds
    with h5py.File('ts/' + folder + '/mean_rgb_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('mean', data=mean_rgb_ds_ts, dtype='f')

    with h5py.File('ts/' + folder + '/norm_mean_rgb_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('x_train', data=x_train_norm_mean, dtype='f')
        f.create_dataset('y_train', data=y_train, dtype='i')

        f.create_dataset('x_validation', data=x_valid_norm_mean, dtype='f')
        f.create_dataset('y_validation', data=y_validation, dtype='i')

        f.create_dataset('x_test', data=x_test_norm_mean, dtype='f')
        f.create_dataset('y_test', data=y_test, dtype='i')

    # calc std
    std_rgb_ds_ts = np.std(x_train_norm, axis=0)

    x_train_norm_mean_std = x_train_norm / std_rgb_ds_ts
    x_valid_norm_mean_std = x_valid_norm / std_rgb_ds_ts
    x_test_norm_mean_std = x_test_norm / std_rgb_ds_ts
    # save std_ds
    with h5py.File('ts/' + folder + '/std_rgb_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('std', data=std_rgb_ds_ts, dtype='f')

    with h5py.File('ts/' + folder + '/norm_mean_std_rgb_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('x_train', data=x_train_norm_mean_std, dtype='f')
        f.create_dataset('y_train', data=y_train, dtype='i')

        f.create_dataset('x_validation', data=x_valid_norm_mean_std, dtype='f')
        f.create_dataset('y_validation', data=y_validation, dtype='i')

        f.create_dataset('x_test', data=x_test_norm_mean_std, dtype='f')
        f.create_dataset('y_test', data=y_test, dtype='i')


def preprocessGrayDataset(folder, dbname):
    """
    Normalization, mean subraction, std for gray images
    """
    with h5py.File('ts/' + folder + '/' + dbname + '.hdf5', 'r') as f:
        x_train = f['x_train']
        y_train = f['y_train']
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_validation = f['x_validation']
        y_validation = f['y_validation']
        x_validation = np.array(x_validation)
        y_validation = np.array(y_validation)

        x_test = f['x_test']
        y_test = f['y_test']
        x_test = np.array(x_test)
        y_test = np.array(y_test)

    print(x_train.shape)
    print(len(y_train))
    # Converting all images to gray
    x_train = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), x_train)))
    x_validation = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), x_validation)))
    x_test = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), x_test)))

    # add plus 1 dimension
    x_train = x_train[:, :, :, np.newaxis]
    x_validation = x_validation[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    # apply normalization by divide every pixel by 255
    x_train_norm = x_train / 255.0
    x_valid_norm = x_validation / 255.0
    x_test_norm = x_test / 255.0
    with h5py.File('ts/' + folder + '/norm_gray_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('x_train', data=x_train_norm, dtype='f')
        f.create_dataset('y_train', data=y_train, dtype='i')

        f.create_dataset('x_validation', data=x_valid_norm, dtype='f')
        f.create_dataset('y_validation', data=y_validation, dtype='i')

        f.create_dataset('x_test', data=x_test_norm, dtype='f')
        f.create_dataset('y_test', data=y_test, dtype='i')

    # mean img calc from training ds
    mean_gray_ds_ts = np.mean(x_train_norm, axis=0)

    # keep data centralized around 0, this will speed up training
    x_train_norm_mean = x_train_norm - mean_gray_ds_ts
    x_valid_norm_mean = x_valid_norm - mean_gray_ds_ts
    x_test_norm_mean = x_test_norm - mean_gray_ds_ts

    # saving mean ds
    with h5py.File('ts/' + folder + '/mean_gray_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('mean', data=mean_gray_ds_ts, dtype='f')

    with h5py.File('ts/' + folder + '/norm_mean_gray_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('x_train', data=x_train_norm_mean, dtype='f')
        f.create_dataset('y_train', data=y_train, dtype='i')

        f.create_dataset('x_validation', data=x_valid_norm_mean, dtype='f')
        f.create_dataset('y_validation', data=y_validation, dtype='i')

        f.create_dataset('x_test', data=x_test_norm_mean, dtype='f')
        f.create_dataset('y_test', data=y_test, dtype='i')

    # calc std from mean
    std_gray_ds_ts = np.std(x_train_norm_mean, axis=0)

    x_train_norm_mean_std = x_train_norm_mean / std_gray_ds_ts
    x_valid_norm_mean_std = x_valid_norm_mean / std_gray_ds_ts
    x_test_norm_mean_std = x_test_norm_mean / std_gray_ds_ts
    # save std_ds
    with h5py.File('ts/' + folder + '/std_gray_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('std', data=std_gray_ds_ts, dtype='f')

    with h5py.File('ts/' + folder + '/norm_mean_std_gray_' + dbname + '.hdf5', 'w') as f:
        f.create_dataset('x_train', data=x_train_norm_mean_std, dtype='f')
        f.create_dataset('y_train', data=y_train, dtype='i')

        f.create_dataset('x_validation', data=x_valid_norm_mean_std, dtype='f')
        f.create_dataset('y_validation', data=y_validation, dtype='i')

        f.create_dataset('x_test', data=x_test_norm_mean_std, dtype='f')
        f.create_dataset('y_test', data=y_test, dtype='i')


def generateAugmentedDataset(folder="aug", dbname='dataset_ts_augmented'):
    mergeDatasets()
    equalizeDatasetByAugmentation(folder + '/' + dbname)
    preprocessRGBDataset(folder, dbname)
    preprocessGrayDataset(folder, dbname)


def generateOriginalDataset(folder="orig", dbname="dataset_ts_orig"):
    mergeDatasetsWithTestDataset(folder + '/' + dbname)
    preprocessRGBDataset(folder, dbname)
    preprocessGrayDataset(folder, dbname)


# preprocessing methods needs to run only once
preproccessTrainingDataset()
preprocessTestDataset()

generateAugmentedDataset()
print("augmented dataset is done")
generateOriginalDataset()
print("original dataset is done")
