#!/usr/bin/env bash

python3 real_time_object_detection_custom.py --prototxt prototxt/dnn_deploy.prototxt --model models/dnn_deploy.caffemodel --labels custom_words.txt --source webcam