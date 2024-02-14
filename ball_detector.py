#https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
import cv2
import numpy as np
from torchvision.models import ResNet
import torch
class BallDetector:
    def __init__(self):
        pass

    def preprocess_frame(self, frame):
        # Convert BGR (OpenCV) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the numpy array frame to a torch tensor and add a batch dimension ([H, W, C] to [C, H, W])
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
        # Normalize pixel values to [0, 1]
        frame_tensor /= 255.0
        return frame_tensor.unsqueeze(0)  # Add batch dimension

    def detect(self, frame, model):
        # Convert BGR (OpenCV) to RGB
        frame_tensor = self.preprocess_frame(frame)
        # print(frame_tensor)
        # Perform detection
        with torch.no_grad():
            prediction = model(frame_tensor)

        # Extract boxes and labels from the prediction
        boxes = prediction[0]['boxes'].detach().numpy()
        labels = prediction[0]['labels'].detach().numpy()
        scores = prediction[0]['scores'].detach().numpy()

        # Adjust the confidence threshold if needed
        confidence_threshold = 0.5  # Lowered to 0.5 for testing
        high_confidence_indices = scores > confidence_threshold
        boxes = boxes[high_confidence_indices]
        labels = labels[high_confidence_indices]

        return boxes, labels