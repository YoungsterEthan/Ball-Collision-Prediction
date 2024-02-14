import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
def display_image_with_boxes(image, boxes, labels):
    # Convert tensor image to PIL for easy display
    img = transforms.ToPILImage()(image).convert("RGB")
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    ax = plt.gca()
    for box, label in zip(boxes, labels):
        x, y, x2, y2 = box
        rect = patches.Rectangle((x, y), x2-x, y2-y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f'Label: {label}', bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis("off")
    plt.show()

# Main program to load an image, run the model, and display the results
def main(image_path):
    image = load_image(image_path)
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
