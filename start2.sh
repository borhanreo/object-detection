#!/usr/bin/env bash

python3.6 real_time_object_detection_google.py --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt --source webcam