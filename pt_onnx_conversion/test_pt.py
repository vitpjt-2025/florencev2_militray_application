import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModelForObjectDetection, AutoProcessor  # Import Florence v2 model

# Load Florence v2 Model
model_name = "microsoft/Florence-2-base-ft"  # Use Florence v2 model name
processor = AutoProcessor.from_pretrained(model_name)  # Preprocessing module
model = AutoModelForObjectDetection.from_pretrained(model_name)  # Load model

# Load the fine-tuned state dict (if any)
model.load_state_dict(torch.load("florence2_lora_quantized_pruned.pt", map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")  # Florence's expected input
    return inputs, np.array(image)

# Draw bounding boxes on the image
def draw_bounding_boxes(image, bboxes, labels, scores, threshold=0.5):
    for bbox, label, score in zip(bboxes, labels, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, bbox)  # Convert to int
        label_text = f"{label} ({score:.2f})"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label text
        cv2.putText(image, label_text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# Run inference
image_path = "inp1.jpg"  # Change to your image path
inputs, resized_image = preprocess_image(image_path)

# Ensure model runs on CPU
with torch.no_grad():
    outputs = model(**inputs)  # Run inference

# Extract Florence outputs
bboxes = outputs["pred_boxes"].cpu().numpy().tolist()
labels = [str(label) for label in outputs["pred_labels"].cpu().numpy().tolist()]
scores = outputs["pred_scores"].cpu().numpy().tolist()

# Draw bounding boxes
output_image = draw_bounding_boxes(resized_image, bboxes, labels, scores)

# Save and display the output image
output_path = "output_detected.jpg"
cv2.imwrite(output_path, output_image)

print(f"Detection output saved at: {output_path}")
