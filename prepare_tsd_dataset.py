import pandas as pd
import os
import cv2

ts_dataset_path = '/GTSDB'
full_path = 'F:/pythonprogramok/TSR_projektmunka'
"""
This will convert every ppm into jpg, and creating seperate annotation files for them in yolov3 format and normalize them

results : jpg images, classes.names, train.txt, test.txt, ts_data.data
train.txt and test.txt contains full paths for created jpg images
classes.names contains the class names
ts_data.data contains full path for train.txt, test.txt and classes.names
"""


def createAnnotations():
    # Defining lists for categories according to the classes ID's
    # Prohibitory category:
    prob = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]

    # Danger category:
    danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    # Mandatory category:
    mand = [33, 34, 35, 36, 37, 38, 39, 40]

    # Other category:
    other = [6, 12, 13, 14, 17, 32, 41, 42]

    # original annoation file : all image with ROI in one file
    ann = pd.read_csv(full_path + ts_dataset_path + '/gt.txt',
                      names=['ImageID',
                             'XMin',
                             'YMin',
                             'XMax',
                             'YMax',
                             'ClassID'],
                      sep=';')

    # Converting into yolov3 format
    ann['CategoryID'] = ''
    ann['center x'] = ''
    ann['center y'] = ''
    ann['width'] = ''
    ann['height'] = ''

    # Getting category's ID according to the class's ID
    ann.loc[ann['ClassID'].isin(prob), 'CategoryID'] = 0
    ann.loc[ann['ClassID'].isin(danger), 'CategoryID'] = 1
    ann.loc[ann['ClassID'].isin(mand), 'CategoryID'] = 2
    ann.loc[ann['ClassID'].isin(other), 'CategoryID'] = 3

    # Calculating bounding box's center in x and y for all rows
    ann['center x'] = (ann['XMax'] + ann['XMin']) / 2
    ann['center y'] = (ann['YMax'] + ann['YMin']) / 2

    # Calculating bounding box's width and height for all rows
    ann['width'] = ann['XMax'] - ann['XMin']
    ann['height'] = ann['YMax'] - ann['YMin']

    # getting new Annotations for yolov3 format
    newAnn = ann.loc[:, ['ImageID',
                         'CategoryID',
                         'center x',
                         'center y',
                         'width',
                         'height']].copy()

    # Changing the current directory to one with images
    os.chdir(full_path + ts_dataset_path)

    # going through all directories
    # this will convert ppm to jpg and create separate annotation for each image, and normalize dataset
    # and files in them from the current directory
    for current_dir, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.ppm'):
                # Reading image and getting its real width and height
                image_ppm = cv2.imread(f)
                h, w = image_ppm.shape[:2]

                image_name = f[:-4]
                sub_Ann = newAnn.loc[newAnn['ImageID'] == f].copy()

                # Normalizing calculated bounding boxes' coordinates
                # according to the real image width and height
                sub_Ann['center x'] = sub_Ann['center x'] / w
                sub_Ann['center y'] = sub_Ann['center y'] / h
                sub_Ann['width'] = sub_Ann['width'] / w
                sub_Ann['height'] = sub_Ann['height'] / h

                resulted_frame = sub_Ann.loc[:, ['CategoryID',
                                                 'center x',
                                                 'center y',
                                                 'width',
                                                 'height']].copy()

                # Checking if there is no any annotations for current image
                if resulted_frame.isnull().values.all():
                    # Skipping this image
                    continue

                path_to_save = ts_dataset_path + '/' + image_name + '.txt'

                # Saving annotation for current img
                resulted_frame.to_csv(full_path + path_to_save, header=False, index=False, sep=' ')

                # convert current img to jpg
                path_to_save = full_path + ts_dataset_path + '/' + image_name + '.jpg'
                cv2.imwrite(path_to_save, image_ppm)


def createClassesNamesTsDataData():
    # generate classes.txt
    with open(full_path + ts_dataset_path + '/classes.txt', 'w') as f:
        f.write('prohibitory' + '\n')
        f.write('danger' + '\n')
        f.write('mandatory' + '\n')
        f.write('other')

    # Defining counter for classes
    c = 0
    # Creating file classes.names from existing one classes.txt
    with open(full_path + ts_dataset_path + '/classes.names', 'w') as names, \
            open(full_path + ts_dataset_path + '/classes.txt', 'r') as txt:
        # Going through all lines in txt file and writing them into classes.names file
        for line in txt:
            names.write(line)
            c += 1

    # generating ts_data.data file
    with open(full_path + ts_dataset_path + '/' + 'ts_data.data', 'w') as data:
        # Number of classes
        data.write('classes = ' + str(c) + '\n')
        # train.txt file path
        data.write('train = ' + full_path + ts_dataset_path + '/' + 'train.txt' + '\n')
        # test.txt file path
        data.write('valid = ' + full_path + ts_dataset_path + '/' + 'test.txt' + '\n')
        # classes.names file path
        data.write('names = ' + full_path + ts_dataset_path + '/' + 'classes.names' + '\n')
        # path where to save weights
        data.write('backup = backup')


def createTestTrainDataset():
    os.chdir(full_path + ts_dataset_path)

    # Defining list to write paths in
    paths = []

    for current_dir, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.jpg'):
                # save into train.txt file
                path_to_save_into_txt_files = full_path + ts_dataset_path + '/' + f
                # Appending the line into the list
                paths.append(path_to_save_into_txt_files + '\n')

    # Slicing first 15% of elements from the list for test
    p_test = paths[:int(len(paths) * 0.15)]

    # Deleting from initial list first 15% of elements
    paths = paths[int(len(paths) * 0.15):]

    # Creating train.txt and test.txt files
    # Creating file train.txt and writing 85% of lines in it
    with open('train.txt', 'w') as train_txt:
        for e in paths:
            # Writing current path at the end of the file
            train_txt.write(e)

    # Creating file test.txt and writing 15% of lines in it
    with open('test.txt', 'w') as test_txt:
        for e in p_test:
            # Writing current path at the end of the file
            test_txt.write(e)


createAnnotations()
createClassesNamesTsDataData()
createTestTrainDataset()
