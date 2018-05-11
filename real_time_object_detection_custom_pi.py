# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --source webcam

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import sys
import argparse
import imutils
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

host = 'http://192.168.0.101:8080/'
url = host + 'shot.jpg'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("--source", required=True,
                help="Source of video stream (webcam/host)")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
                help="minimum probability to filter weak detections")
ap.add_argument("-l", "--labels", required=True,
                help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
rows = open(args["labels"]).read().strip().split("\n")
CLASSES = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
#camera.resolution = (300, 300)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(640, 480))
#rawCapture = PiRGBArray(camera, size=(300, 300))
time.sleep(0.1)

# loop over the frames from the video stream
# for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#     image = frame.array
#     (h, w) = image.shape[:2]
#
#     # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
#     blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1, (224, 224), (104, 117, 123))
#
#     # pass the blob through the network and obtain the detections and predictions
#     net.setInput(blob)
#     detections = net.forward()
#
#     # sort the indexes of the probabilities in descending order (higher
#     # probabilitiy first) and grab the top-5 predictions
#     idxs = np.argsort(detections[0])[::-1][:5]
#
#     # loop over the top-5 predictions and display them
#     for (i, idx) in enumerate(idxs):
#         confidence = detections[0, idx]
#
#         # filter out weak detections by ensuring the `confidence` is
#         # greater than the minimum confidence
#         if confidence > args["confidence"]:
#
#             # draw the top prediction on the input image
#             if i == 0:
#                 text = "Label: {}, {:.2f}%".format(CLASSES[idx], confidence * 100)
#                 cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#             # display the predicted label + associated probability to the console
#             print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, CLASSES[idx], confidence))
#
#     # show the output frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break

detected_objects = []
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            detected_objects.append(label)
            print(label)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
