# Codebase for the Project: Transformer-Based Enemy Detection System
This is a final year project created by authors Cynthia Konar, Sweety Singh and Diya Das under the guidance of Dr. Subitha D in Vellore Insititute of Technology, Chennai. 

The project uses a Trained ViT Florence-v2 Base Machine Learning Model for Detecting Threats in a Military Base Environment.

Dataset link - [https://universe.roboflow.com/capstone2025-mifho/military-base-object-detection](https://universe.roboflow.com/capstone2025-mifho/military-base-object-detection)

## Mini model idea:
![Mini model idea](https://github.com/user-attachments/assets/b3939a91-10fc-4c48-863b-ec0db8d78a8c)
![detections (1)](https://github.com/user-attachments/assets/70602683-74e7-45f1-aa49-ea1ab0cc04a6)

## Diagrams:
![overall_architecture (1)](https://github.com/user-attachments/assets/3dd1f453-fa6f-4881-9131-45d643afe91c)
![florence_architecture (1)](https://github.com/user-attachments/assets/868f85c2-d303-47b4-8e6c-5376546aa272)
![hardware_architecture (1)](https://github.com/user-attachments/assets/db0409be-1b3a-4544-abd2-9a673b9894e7)

## Implementation Images

### GAN Input for Aircrafts
![WhatsApp Image 2025-01-22 at 15 51 41_e3462bb3](https://github.com/user-attachments/assets/ac7657f0-b95a-49d5-8011-ba68f29ef0c4)

### GAN Output for Aircrafts
![WhatsApp Image 2025-02-20 at 22 07 37_48e4c430](https://github.com/user-attachments/assets/9cb141c4-a981-4809-8e38-f25635b0ff5c)

### Testing Florencev2 with laptop camera:
![Output_collage (2)](https://github.com/user-attachments/assets/9ef54e9a-479e-4e4a-96d1-e0d6f768e3f0)
### Testing Yolov11 with laptop camera:
![WhatsApp Image 2025-02-12 at 18 16 00_feb29bc3](https://github.com/user-attachments/assets/097e0f2a-9075-43a9-8f94-5c8b526254ce)
![WhatsApp Image 2025-02-12 at 18 12 34_8d6bd3cb](https://github.com/user-attachments/assets/13f65575-a68a-43fc-a843-a5df9a54b5d9)

## Accuracy
### Florencev2 Model
![image](https://github.com/user-attachments/assets/15540d15-d802-454e-9568-e2a5b715a48d)

## Confusion Matrix
![confusion_matrix (1)](https://github.com/user-attachments/assets/7fecb6ce-3c90-4a27-9bec-3987f0e1f295)

## Epochs vs Loss
![image](https://github.com/user-attachments/assets/a51d6f37-ab0a-4490-810c-25aff73eff32)

Stopped training when the model started overfitting
## Comparison between State of the Art Models Available 
![compare (1)](https://github.com/user-attachments/assets/e9f8659f-46f2-4c71-9367-68c97ffdff30)

### Initial Lab Setup (without camera / lora)
![WhatsApp Image 2025-02-20 at 18 47 42_271fc1c4](https://github.com/user-attachments/assets/11980949-f8d5-425b-8001-03e87f2baf7f)

### Running model on RaspberryPi with image and video input:
![WhatsApp Image 2025-02-20 at 23 19 38_bd326d58](https://github.com/user-attachments/assets/0a7ddc11-063e-4df2-b2d6-12eac4a83ac2)

### Size comparison of different model formats generated:
![WhatsApp Image 2025-02-20 at 23 06 27_dd01137b](https://github.com/user-attachments/assets/9fd69648-b9ed-4125-9f0f-8639e00a05ec)
