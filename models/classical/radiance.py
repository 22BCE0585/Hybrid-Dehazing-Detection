import numpy as np


def recover_radiance(image, transmission, A, t_min=0.1):
    """
    Recover clear image.
    """
    transmission = np.clip(transmission, t_min, 1)

    J = np.zeros_like(image)

    for c in range(3):
        J[:, :, c] = (image[:, :, c] - A[c]) / transmission + A[c]

    return np.clip(J, 0, 1)