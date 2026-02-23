import numpy as np


def estimate_atmospheric_light(image, dark_channel):
    """
    Estimate atmospheric light using brightest pixels in dark channel.
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    top_pixels = int(max(num_pixels * 0.001, 1))

    # Flatten
    dark_vec = dark_channel.reshape(num_pixels)
    image_vec = image.reshape(num_pixels, 3)

    # Indices of brightest dark channel pixels
    indices = dark_vec.argsort()[-top_pixels:]

    # Atmospheric light
    A = np.mean(image_vec[indices], axis=0)

    return A