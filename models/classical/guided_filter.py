import numpy as np
import cv2


def box_filter(img, r):
    """
    Fast box filter using OpenCV.
    """
    return cv2.blur(img, (r, r))


def guided_filter(I, p, r=40, eps=1e-3):
    """
    Guided filter implementation.
    
    I: guidance image (grayscale)
    p: input image (transmission map)
    r: window radius
    eps: regularization parameter
    """

    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I * p, r)

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(I * I, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * I + mean_b

    return q