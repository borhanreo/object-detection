# Realtime Obejct Recognition

Realtime object recognition using the OpenCV 3.3 dnn module + pretrained MobileNetSSD caffemodel.

[![Realtime object recognition](https://img.youtube.com/vi/LGUR4Rn_kWs/0.jpg)](https://www.youtube.com/watch?v=LGUR4Rn_kWs)

## Installation

All the dependencies can be installed using `pip`. Just use the following command from the root directory of the project.

```bash
pip3 install -r requirements.txt
```

**NOTE:** At the time of this writing, `public.py` is broken. If you have a problem installing it, check this [workaround here](https://github.com/C-Aniruddh/realtime_object_recognition/issues/1).

## How to run this script?

There are two options for video source:

 * Webcam
 * Android device running IP Camera (https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en)

To run the script using webcam as source :

```bash
bash start1.sh
```

To run the script using IP Webcam as source, open the `real_time_object_detection.py` and edit the following line to match your host :

```python
host = 'http://192.168.0.101:8080/'
```

Then to run the script using IP as source :

```bash
python3 real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --source web
```

For any questions, create an issue in this repository.

## Make SSD Model

    git clone https://github.com/weiliu89/caffe.git
    git checkout ssd
    
Install requirements from https://code.leftofthedot.com/mamun/caffemodel
    
    cp Makefile.config.example Makefile.config
    make -j8 
    export CAFFE_ROOT=/home/mamun/Idea/env/caffe
    export PYTHONPATH=$CAFFE_ROOT/python:$PROTO_ROOT/python:$PYTHONPATH
    make py
    make test -j8
    make runtest -j8
    
By default, we assume the model is stored in $CAFFE_ROOT/models/VGGNet/
    
    mkdir $CAFFE_ROOT/models/VGGNet/
    wget https://gist.githubusercontent.com/weiliu89/2ed6e13bfd5b57cf81d6/raw/758667b33d1d1ff2ac86b244a662744b7bb48e01/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt -P $CAFFE_ROOT/models/VGGNet/
    
Download VOC2007 and VOC2012 dataset. By default, we assume the data is stored in $HOME/data/
    
    mkdir $HOME/data && cd $HOME/data
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    # Extract the data.
    tar -xvf VOCtrainval_11-May-2012.tar
    tar -xvf VOCtrainval_06-Nov-2007.tar
    tar -xvf VOCtest_06-Nov-2007.tar
    
Create the LMDB file.

    cd $CAFFE_ROOT

Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
    
    ./data/VOC0712/create_list.sh
    
You can modify the parameters in create_data.sh if needed. It will create lmdb files for trainval and test with encoded original image:
- $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
- $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
and make soft links at examples/VOC0712/


    ./data/VOC0712/create_data.sh