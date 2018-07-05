#!/usr/bin/env bash

python3 real_time_object_detection_google.py --prototxt prototxt/bvlc_googlenet.prototxt --model models/bvlc_googlenet.caffemodel --labels labels/synset_words.txt --source webcam