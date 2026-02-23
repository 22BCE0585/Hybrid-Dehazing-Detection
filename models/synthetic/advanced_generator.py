import numpy as np
import cv2
import os
from tqdm import tqdm


def generate_random_depth(h, w):
    noise = np.random.rand(h, w)
    depth = cv2.GaussianBlur(noise, (51, 51), 0)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    return depth


def generate_spatial_beta(h, w):
    beta_noise = np.random.uniform(0.5, 2.0, (h, w))
    beta_map = cv2.GaussianBlur(beta_noise, (101, 101), 0)
    return beta_map


def generate_spatial_atmosphere(h, w):
    A_base = np.random.uniform(0.6, 1.0, 3)

    variation = np.random.uniform(0.8, 1.2, (h, w, 3))
    variation = cv2.GaussianBlur(variation, (101, 101), 0)

    A_map = A_base * variation
    return np.clip(A_map, 0, 1)


def add_illumination_variation(image):
    h, w = image.shape[:2]
    illum = np.random.uniform(0.5, 1.2, (h, w))
    illum = cv2.GaussianBlur(illum, (101, 101), 0)
    image = image * illum[:, :, None]
    return np.clip(image, 0, 1)


def add_sensor_noise(image):
    noise = np.random.normal(0, 0.02, image.shape)
    return np.clip(image + noise, 0, 1)


def generate_dataset(clean_images, output_dir, num_samples=500):

    os.makedirs(os.path.join(output_dir, "hazy"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "transmission"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)

    count = 0

    while count < num_samples:

        img = clean_images[np.random.randint(0, len(clean_images))]
        h, w = img.shape[:2]

        # Apply illumination variation
        img_mod = add_illumination_variation(img)

        depth = generate_random_depth(h, w)
        beta_map = generate_spatial_beta(h, w)

        transmission = np.exp(-beta_map * depth)

        A_map = generate_spatial_atmosphere(h, w)

        hazy = img_mod * transmission[:, :, None] + A_map * (1 - transmission[:, :, None])

        hazy = add_sensor_noise(hazy)

        cv2.imwrite(f"{output_dir}/hazy/{count}.png",
                    (hazy * 255).astype(np.uint8))

        cv2.imwrite(f"{output_dir}/transmission/{count}.png",
                    (transmission * 255).astype(np.uint8))

        cv2.imwrite(f"{output_dir}/clean/{count}.png",
                    (img * 255).astype(np.uint8))

        count += 1