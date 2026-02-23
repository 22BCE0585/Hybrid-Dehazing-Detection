from ultralytics import YOLO
import cv2
import numpy as np


class YOLODetector:

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect(self, image, save_path=None):

        results = self.model(image, verbose=False)

        boxes = results[0].boxes

        num_detections = len(boxes) if boxes is not None else 0
        confidences = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) > 0 else []

        avg_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0

        if save_path:
            annotated = results[0].plot()
            cv2.imwrite(save_path, annotated)

        return num_detections, avg_conf