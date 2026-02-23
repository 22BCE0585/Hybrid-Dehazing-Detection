import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def evaluate(clean_path, classical_path, hybrid_path):

    clean = cv2.imread(clean_path)
    classical = cv2.imread(classical_path)
    hybrid = cv2.imread(hybrid_path)

    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
    classical = cv2.cvtColor(classical, cv2.COLOR_BGR2RGB)
    hybrid = cv2.cvtColor(hybrid, cv2.COLOR_BGR2RGB)

    clean = clean.astype(np.float32) / 255.0
    classical = classical.astype(np.float32) / 255.0
    hybrid = hybrid.astype(np.float32) / 255.0

    psnr_classical = psnr(clean, classical, data_range=1.0)
    psnr_hybrid = psnr(clean, hybrid, data_range=1.0)

    ssim_classical = ssim(clean, classical, channel_axis=2, data_range=1.0)
    ssim_hybrid = ssim(clean, hybrid, channel_axis=2, data_range=1.0)

    print("----- Evaluation Results -----")
    print("Classical PSNR:", psnr_classical)
    print("Hybrid PSNR:", psnr_hybrid)
    print("Classical SSIM:", ssim_classical)
    print("Hybrid SSIM:", ssim_hybrid)