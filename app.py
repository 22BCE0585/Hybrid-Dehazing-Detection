import streamlit as st
import numpy as np
import cv2
from streamlit_image_comparison import image_comparison
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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

st.set_page_config(page_title="Hybrid Dehazing System", layout="wide")

st.title("Hybrid Physics-Informed Image Dehazing Framework")

st.sidebar.header("Configuration")

haze_strength = st.sidebar.slider("Haze Intensity", 0.8, 2.2, 1.6)
show_trans = st.sidebar.checkbox("Show Transmission Maps", True)
show_architecture = st.sidebar.checkbox("Show Architecture Diagram", False)
enable_quality_eval = st.sidebar.checkbox("Enable PSNR / SSIM Evaluation", False)

def normalize_display(img):
    img = np.clip(img, 0, 1)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def color_map(img):
    img = normalize_display(img)
    img = (img * 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if enable_quality_eval:
    gt_file = st.file_uploader("Upload Ground Truth Image", type=["jpg", "png"])
else:
    gt_file = None

if uploaded_file:

    with st.spinner("Processing Image..."):

        clean = load_image(uploaded_file)
        h, w = clean.shape[:2]

        # Generate haze
        clean_mod = add_illumination_variation(clean)
        depth = generate_random_depth(h, w)
        beta_map = generate_spatial_beta(h, w) * haze_strength
        transmission = np.exp(-beta_map * depth)
        A_map = generate_spatial_atmosphere(h, w)

        hazy = clean_mod * transmission[:, :, None] + A_map * (1 - transmission[:, :, None])
        hazy = add_sensor_noise(hazy)
        hazy = np.clip(hazy, 0, 1)

        # Classical
        dark = get_dark_channel(hazy)
        A = estimate_atmospheric_light(hazy, dark)
        raw_trans = estimate_transmission(hazy, A)

        gray = cv2.cvtColor((hazy * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        classical_trans = guided_filter(gray, raw_trans)
        classical_output = recover_radiance(hazy, classical_trans, A)

        # Hybrid
        refiner = CNNRefiner("models/cnn_refiner/checkpoints/best_model.pth")
        refined_trans, A = refiner.refine(hazy)
        hybrid_output = recover_radiance(hazy, refined_trans, A)

        # Detection
        detector = YOLODetector("yolov8n.pt")
        hazy_conf = detector.detect((hazy * 255).astype(np.uint8))[1]
        classical_conf = detector.detect((classical_output * 255).astype(np.uint8))[1]
        hybrid_conf = detector.detect((hybrid_output * 255).astype(np.uint8))[1]

    st.subheader("Interactive Comparison")

    image_comparison(
        img1=(hazy * 255).astype(np.uint8),
        img2=(hybrid_output * 255).astype(np.uint8),
        label1="Hazy",
        label2="Hybrid"
    )

    if show_trans:
    
        st.subheader("Transmission Maps")

        col1, col2 = st.columns(2)
        col1.image(color_map(classical_trans), caption="Classical", width="stretch")
        col2.image(color_map(refined_trans), caption="Hybrid", width="stretch")

    st.subheader("Detection Confidence")

    improvement = hybrid_conf - classical_conf

    st.markdown(f"""
    - **Hazy:** {hazy_conf:.4f}
    - **Classical:** {classical_conf:.4f}
    - **Hybrid:** {hybrid_conf:.4f}
    - **Hybrid Gain:** {improvement:.4f}
    """)

    if gt_file:
        gt = load_image(gt_file)

        psnr_val = psnr(gt, hybrid_output)
        ssim_val = ssim(gt, hybrid_output, channel_axis=2)

        st.subheader("Quality Metrics (vs Ground Truth)")
        st.write(f"PSNR: {psnr_val:.2f}")
        st.write(f"SSIM: {ssim_val:.4f}")

    st.download_button(
        "Download Hybrid Output",
        data=cv2.imencode('.png', (hybrid_output * 255).astype(np.uint8))[1].tobytes(),
        file_name="hybrid_output.png"
    )

if show_architecture:
    st.subheader("System Architecture")
    st.image("architecture.png", width="stretch")