import cv2
import torchvision.models.detection as detection
from PIL import Image
import torch
from ball_detector import BallDetector
from kalmanfilter import KalmanFilter
import numpy as np
from imagedetection import display_image_with_boxes
import torchvision.transforms as transforms
##############################################################
def load_model():
    # Load the model outside of the read_frame function
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model
###########################################################

cap = cv2.VideoCapture("322 - Elastic collissions of the balls.mp4")

# Load detector
detector = BallDetector()
# Load Kalman filter to predict the trajectory
kf = KalmanFilter()
kf2 = KalmanFilter()
model = load_model()
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    boxes, labels = detector.detect(frame, model)
    for box in boxes:
        x, y, x2, y2 = box
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)
        predicted = kf.predict(cx, cy)
        cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)

    print(boxes)
    print(labels)
    display_image_with_boxes(frame, boxes, labels)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break