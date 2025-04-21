import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

# Load ONNX model
onnx_path = "dinov2_florence2_od.onnx"
ort_session = ort.InferenceSession(onnx_path)

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((518, 518)),  # Match input size
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).numpy(), np.array(image)  # Return numpy image too

# Draw bounding boxes on image
def draw_bounding_boxes(image, bboxes, labels, scores, threshold=0.5):
    for bbox, label, score in zip(bboxes, labels, scores):
        if score < threshold:  # Ignore low confidence scores
            continue
        x1, y1, x2, y2 = map(int, bbox)  # Convert to int
        label_text = f"{label} ({score:.2f})"
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put label
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# Run inference
image_path = "inp4.jpg"
input_tensor, original_image = preprocess_image(image_path)

# Run ONNX model
outputs = ort_session.run(None, {"input": input_tensor})
bboxes, labels, scores = outputs  

# Convert labels to string
labels = [str(label) for label in labels]

# Resize to (518, 518)
original_image_resized = cv2.resize(original_image, (518, 518))

# Draw boxes on image
output_image = draw_bounding_boxes(original_image_resized, bboxes, labels, scores)


# Save and display image
cv2.imwrite("output.jpg", output_image)
cv2.imshow("Detected Objects", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
