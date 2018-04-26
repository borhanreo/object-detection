# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --source webcam

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import sys
import argparse
import imutils
import time
import cv2
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

# initialize the video stream, allow the cammera sensor to warmup,
print("[INFO] starting video stream...")

if args["source"] == "webcam":
    vs = cv2.VideoCapture(0)

time.sleep(2.0)

detected_objects = []
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    if args["source"] == "webcam":
        ret, frame = vs.read()
    else:
        imgResp = urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgNp, -1)

    frame = imutils.resize(frame, width=800)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 1, (224, 224), (104, 117, 123))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions
    idxs = np.argsort(detections[0])[::-1][:5]

    # loop over the top-5 predictions and display them
    for (i, idx) in enumerate(idxs):
        confidence = detections[0, idx]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:

            # draw the top prediction on the input image
            if i == 0:
                text = "Label: {}, {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # display the predicted label + associated probability to the console
            print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, CLASSES[idx], confidence))

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
