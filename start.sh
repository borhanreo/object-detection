#!/usr/bin/env bash

python3.6 real_time_object_detection_custom.py --prototxt dnn_deploy.prototxt --model dnn_deploy.caffemodel --labels custom_words.txt --source webcam