import numpy as np


def generate_transmission_map(image_shape, beta_range=(0.6, 1.8)):
    """
    Generate synthetic transmission map based on depth simulation.
    """
    h, w = image_shape[:2]

    # Simulated depth map (simple gradient)
    depth = np.tile(np.linspace(0, 1, w), (h, 1))

    beta = np.random.uniform(*beta_range)

    transmission = np.exp(-beta * depth)

    return transmission


def add_haze(image, transmission, A=None):
    """
    Add haze to a clean image.
    """
    if A is None:
        A = np.random.uniform(0.7, 1.0, size=3)

    hazy = np.zeros_like(image)

    for c in range(3):
        hazy[:, :, c] = image[:, :, c] * transmission + A[c] * (1 - transmission)

    return np.clip(hazy, 0, 1), A