import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# Load the pre-trained model
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to load and transform an image
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    return img

# Function to display image and bounding boxes
def display_image_with_boxes(frame, boxes, labels):
    for box, label in zip(boxes, labels):
        x, y, x2, y2 = box
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)
        cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 4)



# Main program to load an image, run the model, and display the results
def main(image_path):
    image = load_image(image_path)
    print(image)
    with torch.no_grad():
        prediction = model([image])

    # Extract boxes and labels from the prediction
    boxes = prediction[0]['boxes'].detach().numpy()
    labels = prediction[0]['labels'].detach().numpy()

    # Filter out low confidence detections
    high_confidence_indices = prediction[0]['scores'] > 0.8  # Threshold can be adjusted
    boxes = boxes[high_confidence_indices]
    labels = labels[high_confidence_indices]

    display_image_with_boxes(image, boxes, labels)

if __name__ == "__main__":
    image_path = 'Screenshot (63).png'  # Update this path to your image
    main(image_path)
