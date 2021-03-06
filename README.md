# Realtime Obejct Recognition

Realtime object recognition using the OpenCV 3.3 dnn module + pretrained MobileNetSSD caffemodel.

[![Realtime object recognition](https://img.youtube.com/vi/LGUR4Rn_kWs/0.jpg)](https://www.youtube.com/watch?v=LGUR4Rn_kWs)

## Installation
All the dependencies can be installed using `pip`. Just use the following command from the root directory of the project.

```bash
pip3 install -r requirements.txt
pip3 install -U scikit-image
pip3 install -U cython
sudo apt-get install python3-tk
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```
if pip will python3 then you need to must use pip not pip3
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

For GPU 

    git clone https://github.com/weiliu89/caffe.git
    cd caffe
    git checkout ssd
    
For CPU

    git clone https://github.com/intel/caffe.git
    sudo apt-get install cmake
    cd caffe
    
Install requirements from https://github.com/mdmamunhasan/caffemodel
    
    cp Makefile.config.example Makefile.config
    make -j8 
    export CAFFE_ROOT=/home/mamun/Idea/env/caffe
    export PYTHONPATH=$CAFFE_ROOT/python:$PROTO_ROOT/python:$PYTHONPATH
    make py
    make test -j8
    make runtest -j8
    cd $CAFFE_ROOT/python
    sudo pip install -r requirements.txt
    sudo apt-get install cuda
    
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
    
Train your model and evaluate the model on the fly.

    # It will create model definition files and save snapshot models in:
    #   - $CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300/
    # and job file, log file, and the python script in:
    #   - $CAFFE_ROOT/jobs/VGGNet/VOC0712/SSD_300x300/
    # and save temporary evaluation results in:
    #   - $HOME/data/VOCdevkit/results/VOC2007/SSD_300x300/
    # It should reach 77.* mAP at 120k iterations.
    
    sudo nano $CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300 solver.prototxt
    
Replace solver_mode from GPU to CPU

    python examples/ssd/ssd_pascal.py
    
    
If you don't have time to train your model, you can download a pre-trained model models_VGGNet_VOC0712_SSD_300x300.

Evaluate the most recent snapshot.

    ./build/tools/caffe train -solver models/intel_optimized_models/ssd/VGGNet/VOC0712/SSD_300x300/solver.prototxt \
    -weights models/intel_optimized_models/ssd/VGGNet/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel

If you would like to test a model you trained, you can do:

    python examples/ssd/score_ssd_pascal.py

If you would like to attach a webcam to a model you trained, you can do:

    python examples/ssd/ssd_pascal_webcam.py
    
**Note**: press esc to stop.
    
## Install Cuda

Download cuda from https://developer.nvidia.com/cuda-downloads

    sudo dpkg -i cuda-repo-ubuntu1710-9-2-local_9.2.88-1_amd64.deb
    sudo apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install cuda
    
Force install
    
    sudo dpkg -i --force-overwrite /var/cuda-repo-9-2-local/./nvidia-396_396.26-0ubuntu1_amd64.deb
    sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda
    
Update cuda dir in Makefile.config 

    CUDA_DIR := /usr/local/cuda-9.2

## Uninstall Cuda

Uninstall just nvidia-cuda-toolkit

    sudo apt-get remove nvidia-cuda-toolkit

Uninstall nvidia-cuda-toolkit and it's dependencies

    sudo apt-get remove --auto-remove nvidia-cuda-toolkit
    
remove the CUDA files in /usr/local/cuda-9.2 

    sudo apt-key del 7fa2af80
    sudo rm -R /var/cuda-repo-9-2-local
    sudo apt-key --keyring /tmp/test list
