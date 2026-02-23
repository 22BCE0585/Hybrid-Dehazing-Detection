import numpy as np
import cv2


def estimate_transmission(image, A, omega=0.95, patch_size=15):
    """
    Estimate transmission map.
    """
    norm_image = image / A

    min_channel = np.min(norm_image, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    transmission = 1 - omega * cv2.erode(min_channel, kernel)

    return transmission