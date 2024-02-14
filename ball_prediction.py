import cv2
from ball_detector import BallDetector
from kalmanfilter import KalmanFilter
import numpy as np

cap = cv2.VideoCapture("322 - Elastic collissions of the balls.mp4")

# Load detector
low_green = np.array([25, 52, 72])
high_green = np.array([85, 255, 255])  # Adjust these as needed
low_purple = np.array([130, 50, 50])  # Example starting values for purple
high_purple = np.array([160, 255, 255])  # Adjust these as needed
pBall = BallDetector(low_purple, high_purple)
gBall = BallDetector(low_green, high_green)

# Load Kalman filter to predict the trajectory
kf = KalmanFilter()
kf2 = KalmanFilter()

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    purple_bbox = pBall.detect(frame)
    green_bbox = gBall.detect(frame)
    x, y, x2, y2 = purple_bbox
    i, j, i2, j2 = green_bbox
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)

    ci = int((i + i2) / 2)
    cj = int((j + j2) / 2)

    predicted = kf.predict(cx, cy)
    predicted2 = kf2.predict(ci, cj)
    cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 4)
    cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 4)
    cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)

    cv2.circle(frame, (ci, cj), 20, (0, 0, 255), 4)
    cv2.circle(frame, (predicted2[0], predicted2[1]), 20, (255, 0, 0), 4)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(100)
    if key == 27:
        break