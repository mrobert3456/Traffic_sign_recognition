# Traffic Sign Recognition


![project_video_clip](./data/output.gif)


# ⚙ How it works

1. **Traffic sign detection** using [YoloV3](https://pjreddie.com/darknet/yolo/), trained with [GTSDB dataset](https://benchmark.ini.rub.de/gtsdb_news.html) using [Darknet framework](https://pjreddie.com/darknet/)
2. **Traffic sign detection** trained with [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_news.html) using Convolutional Neural Network


For detailed description of how it works, please check out my publication on IEEE:
https://ieeexplore.ieee.org/abstract/document/10158539
---
# 📦 Installation

## This Repository

Download this repository by running:

```
git clone https://github.com/mrobert3456/Traffic_sign_recognition.git
cd Traffic_sign_recognition
```

## ⚡ Software Dependencies

This project utilizes the following packages:

* Python 3
* OpenCV 2
* Matplotlib
* Numpy
* Pandas
* h5py
* [Tensorflow-gpu](https://www.tensorflow.org/install/pip)
* [Filterpy](https://filterpy.readthedocs.io/en/latest/)

To setup the environment,you need to install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), then run the following commands:
```
conda env create -f environment.yaml
conda activate GPU_ENV
```


# 🚀 Usage

## Traffic sign recognition
1. Before you run the preprocessing methods and training, create a folder called **GTSRB** in the **root** folder
2. Download the **GTSRB** dataset, which contains the following zip files, that you need to unzip in the **GTSRB** folder:
   * GTSRB_Final_Test_Images.zip
   * GTSRB_Final_Training_Images.zip
3. Unzip the **GTSRB_Final_Test_GT.zip** file into the **GTSRB/Final_Test/Images** folder

4. In the **root** folder, create a folder called **'ts'**
   * Inside the **'ts'** folder create two subfolders **'aug'** and **'orig'**
      * This is where the preprocessing and the training results will be saved

To run the preprocessing for the **GTSRB** dataset, just run the following command:
```commandline
python preprocess_tsr_dataset.py
```

To create and train the recognition model, run the following jupyter notebook file:
```commandline
Build_Train_TSR.ipynb
```

## Traffic sign detection

1. Create a folder called **GTSDB** under the root folder
2. Download the **GTSDB** dataset, unzip the **FullIJCNN2013.zip** file under the **GTSDB** folder
3. To run the preprocess method for the GTSDB dataset, just run the following command:

```commandline
python prepare_tsd_dataset.py
```

After the process is finished, the following files will be generated:
   * test.txt
   * train.txt
   * classes.names 
   * ts_data.data 

The **classes.names**, **ts_data.data**, **yolov3_ts_test.cfg** and the **yolov3_ts_train.cfg** files needs to be placed under the **darknet-master/build/darknet/x64/cfg** folder, where you installed the Darknet framework
   * **yolov3_ts_test.cfg** and the **yolov3_ts_train.cfg** files can be found under **'ts'** folder


4. Create a **'weights'** folder under **darknet-master/build/darknet/x64**
   * Copy the **darknet53.conv.74** file under it

To start the training process run the following command:
```commandline
darknet.exe detector train cfg\ts_data.data 
cfg\yolov3_ts_train.cfg weights\darknet53.conv.74
```

The necessary model and weight files can be found here: https://drive.google.com/file/d/1nLjbzzL77rTNSduE5hl7cByP0lgvai2i/view?usp=sharing

These files need to be placed under ``ts`` folder 
```
python detect_recogn.py <input_file.mp4> <output_file.mp4>
```

The output file will be saved as ```output.mp4```

