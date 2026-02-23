import os
import cv2
import numpy as np

from models.synthetic.advanced_generator import (
    generate_random_depth,
    generate_spatial_beta,
    generate_spatial_atmosphere,
    add_illumination_variation,
    add_sensor_noise
)

from models.classical.dark_channel import get_dark_channel
from models.classical.atmospheric_light import estimate_atmospheric_light
from models.classical.transmission import estimate_transmission
from models.classical.guided_filter import guided_filter
from models.classical.radiance import recover_radiance

from models.cnn_refiner.inference import CNNRefiner
from models.detection.detect import YOLODetector


# -------------------------------
# Utility Functions
# -------------------------------

def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def generate_complex_haze(clean_image, haze_strength=1.6):
    h, w = clean_image.shape[:2]

    clean_mod = add_illumination_variation(clean_image)

    depth = generate_random_depth(h, w)
    beta_map = generate_spatial_beta(h, w) * haze_strength

    transmission = np.exp(-beta_map * depth)

    A_map = generate_spatial_atmosphere(h, w) * 1.1
    A_map = np.clip(A_map, 0, 1)

    hazy = clean_mod * transmission[:, :, None] + A_map * (1 - transmission[:, :, None])
    hazy = add_sensor_noise(hazy)
    hazy = np.clip(hazy, 0, 1)

    return hazy


def classical_dehaze(hazy_image):
    dark = get_dark_channel(hazy_image)
    A = estimate_atmospheric_light(hazy_image, dark)
    raw_trans = estimate_transmission(hazy_image, A)

    gray = cv2.cvtColor((hazy_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0

    refined_trans = guided_filter(gray, raw_trans)
    output = recover_radiance(hazy_image, refined_trans, A)

    return output


def hybrid_dehaze(hazy_image, refiner):
    refined_trans, A = refiner.refine(hazy_image)
    output = recover_radiance(hazy_image, refined_trans, A)
    return output

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":

    # -------- Configuration --------
    detection_folder = "data/raw/detection_set"
    haze_strength = 1.6   # Adjust between 1.0 – 2.0 for evaluation

    print("Initializing models...")
    detector = YOLODetector("yolov8n.pt")
    refiner = CNNRefiner("models/cnn_refiner/checkpoints/best_model.pth")

    clean_conf_list = []
    hazy_conf_list = []
    classical_conf_list = []
    hybrid_conf_list = []

    print("Processing images...\n")

    for filename in os.listdir(detection_folder):

        image_path = os.path.join(detection_folder, filename)

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        clean = load_image(image_path)

        # --- Generate Hazy Image ---
        hazy = generate_complex_haze(clean, haze_strength)

        # --- Classical Dehazing ---
        classical_output = classical_dehaze(hazy)

        # --- Hybrid Dehazing ---
        hybrid_output = hybrid_dehaze(hazy, refiner)

        # --- Convert for YOLO ---
        clean_bgr = cv2.imread(image_path)
        hazy_bgr = (hazy * 255).astype(np.uint8)
        classical_bgr = (classical_output * 255).astype(np.uint8)
        hybrid_bgr = (hybrid_output * 255).astype(np.uint8)

        # --- Detection ---
        _, clean_conf = detector.detect(clean_bgr)
        _, hazy_conf = detector.detect(hazy_bgr)
        _, classical_conf = detector.detect(classical_bgr)
        _, hybrid_conf = detector.detect(hybrid_bgr)

        clean_conf_list.append(clean_conf)
        hazy_conf_list.append(hazy_conf)
        classical_conf_list.append(classical_conf)
        hybrid_conf_list.append(hybrid_conf)

        print(f"{filename} processed.")

    # -------- Final Results --------
    print("\n==============================")
    print("   Average Detection Confidence")
    print("==============================")

    print(f"Clean     : {np.mean(clean_conf_list):.4f}")
    print(f"Hazy      : {np.mean(hazy_conf_list):.4f}")
    print(f"Classical : {np.mean(classical_conf_list):.4f}")
    print(f"Hybrid    : {np.mean(hybrid_conf_list):.4f}")

    hybrid_gain = np.mean(hybrid_conf_list) - np.mean(classical_conf_list)

    print("\nHybrid Improvement Over Classical: "
        f"{hybrid_gain:.4f}")