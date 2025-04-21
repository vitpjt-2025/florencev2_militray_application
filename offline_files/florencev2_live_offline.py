import cv2
import torch
import re
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

# Constants
MODEL_PATH = "local_models/florence2-lora"
BASE_MODEL_PATH = "local_models/Florence-2-base-ft"  # Ensure Florence-2-base-ft is downloaded here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Florence-2 processor and model from local paths
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True).to(DEVICE)

peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH).to(DEVICE)
print("PEFT Model loaded on:", next(peft_model.parameters()).device)

# Define class labels and assign random colors
CLASSES = ['Aircraft', 'Camouflage', 'Drone', 'Fire', 'Grenade', 'Gun', 'Hand-Gun', 'Knife',
           'Military-Vehicle', 'Missile', 'Non-Pointing-Gun', 'Person', 'Pistol', 'Pointing-Gun',
           'Rifle', 'Smoke', 'Soldier']
CLASS_COLORS = {cls: tuple(np.random.randint(0, 255, 3).tolist()) for cls in CLASSES}

def parse_output(raw_output, img_width, img_height):
    """ Extract bounding boxes and scale coordinates to match image dimensions. """
    pattern = r"(\w+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    detections = []

    matches = re.findall(pattern, raw_output)
    for match in matches:
        label, x1, y1, x2, y2 = match
        x1 = int(int(x1) / 999 * img_width)
        y1 = int(int(y1) / 999 * img_height)
        x2 = int(int(x2) / 999 * img_width)
        y2 = int(int(y2) / 999 * img_height)

        detections.append({"label": label, "bbox": (x1, y1, x2, y2)})

    return detections

def predict_image(image):
    """ Run inference with Florence-2 LoRA model and extract bounding boxes. """
    prompt = '<OD>'  # Object Detection Task Prompt

    img_height, img_width, _ = image.shape
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Process input
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(DEVICE)

    # Generate output
    with torch.no_grad():
        generated_ids = peft_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )

    # Decode output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Extract class names using regex
    pattern = r"(\w+)<loc_\d+><loc_\d+><loc_\d+><loc_\d+>"
    detected_classes = re.findall(pattern, generated_text)

    for obj in set(detected_classes):  
        print(f"Object Detected: {obj}")

    return parse_output(generated_text, img_width, img_height)

# Initialize webcam
videoCap = cv2.VideoCapture(0)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    detections = predict_image(frame)

    for detection in detections:
        label = detection["label"]
        x1, y1, x2, y2 = detection["bbox"]
        color = CLASS_COLORS.get(label, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Florence-2 Object Detection", frame)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

videoCap.release()
cv2.destroyAllWindows()
