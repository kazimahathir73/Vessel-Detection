from ultralytics import YOLO
import os
import torch

YOLO_MODEL_PATH = "../yolo/yolov7_best_1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = YOLO(YOLO_MODEL_PATH)

def detect_vessels(image_path, output_folder="temp", conf_threshold=0.25):
    os.makedirs(output_folder, exist_ok=True)
    
    results = yolo_model.predict(image_path, conf=conf_threshold, save=True)
    output_path = os.path.join("runs/detect/exp", os.path.basename(image_path))
    print(f"[YOLOv7] Detection result saved at: {output_path}")
    return output_path

if __name__ == "__main__":
    input_path = "temp/image1.jpg"
    detect_vessels(input_path)
