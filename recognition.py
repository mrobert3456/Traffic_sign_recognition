# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 09:13:35 2021

@author: mrobe
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import cv2
import io

from keras.models import load_model

from timeit import default_timer as timer

# loading network
model = load_model('ts' + '/' + 'model_ts_rgb.h5')

# loading trained weights
model.load_weights('ts' + '/' + 'w_1_ts_rgb_255_mean_std.h5')

# loading class names
labels = pd.read_csv('osztalyok.csv', sep=',')

# Converting into Numpy array
labels = np.array(labels.loc[:, 'SignName']).flatten()

# open dataset
with h5py.File('ts' + '/' + 'mean_rgb_dataset_ts.hdf5', 'r') as f:
    # Extracting saved array for Mean Image
    mean_rgb = f['mean']  # HDF5 dataset

    # Converting it into Numpy array
    mean_rgb = np.array(mean_rgb)  # Numpy arrays

# Window to show current view
cv2.namedWindow('Current view', cv2.WINDOW_NORMAL)

# Window to show classification result
cv2.namedWindow('Classified as', cv2.WINDOW_NORMAL)

camera = cv2.VideoCapture("test4.mp4")

# Defining counter for FPS
counter = 0

# Starting timer for FPS
# Getting current time point in seconds
fps_start = timer()

# Creating image with black background
temp = np.zeros((720, 1280, 3), np.uint8)

# processing frames
while camera.isOpened():
    _, frame_bgr = camera.read()

    """
    Start of:
    Detecting object
    """

    # Converting caught frame to HSV colour space
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Applying mask with founded boundary numbers
    mask = cv2.inRange(frame_hsv, (0, 130, 80), (180, 255, 255))

    # Finding contours
    # All found contours are placed into a list
    # Every individual contour is a Numpy array of (x, y) coordinates,
    # that represent boundary points of detected object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Sorting contours from biggest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    """
    End of:
    Detecting object
    """
    """
    Start of:
    Classifying detected object
    """

    # If any contour is found, extracting coordinates of the biggest one
    if contours:
        # Getting rectangle coordinates and spatial size of the biggest contour
        # Function 'cv2.boundingRect()' returns an approximate rectangle,
        # that covers the region around found contour
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])

        # Drawing obtained rectangle on the current  frame
        cv2.rectangle(frame_bgr, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      (230, 161, 0), 3)

        # Putting text above rectangle
        cv2.putText(frame_bgr, 'Detected', (x_min - 5, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 161, 0), 2)

        """
        Start of:
        Cutting detected fragment
        """

        # Cutting detected fragment from BGR frame
        cut_fragment_bgr = frame_bgr[y_min + int(box_height * 0.1):
                                     y_min + box_height - int(box_height * 0.1),
                           x_min + int(box_width * 0.1):
                           x_min + box_width - int(box_width * 0.1)]

        """
        End of:
        Cutting detected fragment
        """

        """
        Start of:
        Preprocessing caught frame
        """

        # Swapping channels from BGR to RGB
        frame_rgb = cv2.cvtColor(cut_fragment_bgr, cv2.COLOR_BGR2RGB)

        # Resizing frame to 48 by 48 pixels size
        frame_rgb = cv2.resize(frame_rgb,
                               (48, 48),
                               interpolation=cv2.INTER_CUBIC)

        # Implementing normalization by dividing image's pixels on 255.0
        frame_rgb_255 = frame_rgb / 255.0

        # Implementing normalization by subtracting Mean Image
        frame_rgb_255_mean = frame_rgb_255 - mean_rgb

        # Extending dimension from (height, width, channels)
        # to (1, height, width, channels)
        frame_rgb_255_mean = frame_rgb_255_mean[np.newaxis, :, :, :]

        """
        End of:
        Preprocessing caught frame
        """

        """
        Start of:
        Implementing forward pass
        """

        # Testing RGB Traffic Signs model trained on dataset:
        # dataset_ts_rgb_255_mean.hdf5
        # Caught frame is preprocessed in the same way
        # Measuring classification time
        start = timer()
        scores = model.predict(frame_rgb_255_mean)
        end = timer()

        # Scores are given as 43 numbers of predictions for each class
        # Getting index of only one class with maximum value
        prediction = np.argmax(scores)

        """
        End of:
        Implementing forward pass
        """

        """
        Start of:
        Showing OpenCV windows
        """

        # Showing current view from camera in Real Time
        # Pay attention! 'cv2.imshow' takes images in BGR format
        cv2.imshow('Current view', frame_bgr)

        # Showing cut fragment
        #  cv2.imshow('Cut fragment', cut_fragment_bgr)

        # Changing background to BGR(230, 161, 0)
        # B = 230, G = 161, R = 0
        temp[:, :, 0] = 230
        temp[:, :, 1] = 161
        temp[:, :, 2] = 0

        # Adding text with current label
        cv2.putText(temp, labels[int(prediction)], (100, 200),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Adding text with obtained confidence score to image with label
        cv2.putText(temp, 'Score : ' + '{0:.5f}'.format(scores[0][prediction]),
                    (100, 450), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255),
                    4, cv2.LINE_AA)

        # Showing classification result
        cv2.imshow('Classified as', temp)
        # print('Prediction: '+labels[int(prediction)]+'   Score: '+str(scores[0][prediction]))

        """
        End of:
        Showing OpenCV windows
        """

    # If no contour is found, showing OpenCV windows with information
    else:
        # Showing current view from camera in Real Time
        # Pay attention! 'cv2.imshow' takes images in BGR format
        cv2.imshow('Current view', frame_bgr)

        # Changing background to BGR(230, 161, 0)
        # B = 230, G = 161, R = 0
        temp[:, :, 0] = 230
        temp[:, :, 1] = 161
        temp[:, :, 2] = 0

        # Adding text with information
        cv2.putText(temp, 'No object', (100, 450),
                    cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)

        # Showing information in prepared OpenCV windows
        cv2.imshow('Classified as', temp)

    """
    End of:
    Classifying detected object
    """

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
