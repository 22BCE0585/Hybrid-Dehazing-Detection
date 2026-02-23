import cv2
import numpy as np


def get_dark_channel(image, patch_size=15):
    """
    Compute the dark channel of the image.
    
    Args:
        image: RGB image normalized to [0,1]
        patch_size: local patch size
        
    Returns:
        dark_channel image
    """
    # Minimum across RGB channels
    min_channel = np.min(image, axis=2)

    # Apply minimum filter (erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel