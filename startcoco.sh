#!/usr/bin/env bash

python3 real_time_object_detection_coco.py --prototxt prototxt/VGG_coco_SSD_300x300_deploy.prototxt --model models/VGG_coco_SSD_300x300_iter_400000.caffemodel --labels labels/coco_words.json --source webcam