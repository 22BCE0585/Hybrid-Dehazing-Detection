Hybrid Physics-Informed Image Dehazing and Object Detection Framework
Overview

This project presents a hybrid image dehazing framework that integrates:

Classical Dark Channel Prior (DCP)

Spatially varying atmospheric scattering simulation

Residual CNN-based transmission refinement

YOLOv8-based object detection robustness evaluation

Interactive Streamlit demonstration interface

The system combines physics-based atmospheric modeling with deep learning to enhance image restoration quality and improve object detection performance under severe haze conditions.

Problem Statement

Haze significantly degrades:

Image visibility

Structural clarity

Object detection confidence

Downstream perception systems

Traditional dehazing approaches:

Assume uniform atmospheric light

Assume global scattering coefficients

Fail under spatially varying haze

Deep learning approaches:

Require large real-world paired datasets

Lack physical interpretability

May produce unrealistic enhancements

This project proposes a hybrid solution that merges physical modeling with learned residual correction.

Key Contributions

Spatially varying atmospheric haze simulation

Physics-based Dark Channel Prior baseline

Residual CNN for transmission refinement

Multi-image statistical detection evaluation

Robustness study under varying haze intensities

Interactive web-based demonstration system

System Architecture

The overall pipeline follows:

Synthetic haze generation

Classical DCP restoration

Residual CNN refinement

Radiance reconstruction

Object detection evaluation

Architecture Diagram

Atmospheric Scattering Model

The system is based on the atmospheric scattering equation:

I(x) = J(x)t(x) + A(1 − t(x))

Where:

I(x) → Observed hazy image

J(x) → Clean scene radiance

t(x) → Transmission map

A → Atmospheric light

The hybrid model learns a residual correction:

t_refined(x) = t_classical(x) + Δt(x)

Where Δt(x) represents the learned correction from the CNN.

Core Components
1. Haze Simulation Module

Implements spatially varying atmospheric scattering with:

Random depth map generation

Spatial beta (scattering coefficient) map

Spatial atmospheric light modeling

Illumination variation

Sensor noise simulation

This module enables controlled robustness testing under different haze intensities.

2. Classical Dehazing (Dark Channel Prior)

Pipeline:

Dark channel computation

Atmospheric light estimation

Transmission estimation

Guided filter refinement

Scene radiance recovery

Provides a physics-based baseline restoration method.

3. Hybrid Residual CNN

A lightweight encoder-decoder network that learns a residual correction to the classical transmission map.

Final transmission:

t_refined(x) = t_classical(x) + Δt(x)

Benefits:

Better edge preservation

Improved local consistency

Reduced over-smoothing

Enhanced structural fidelity

4. Detection Evaluation Module

Uses pretrained YOLOv8 to evaluate:

Object count

Average detection confidence

Robustness under haze

Hybrid vs classical improvement

The detector serves as a downstream robustness evaluation tool.

Experimental Results Summary

Under increasing haze intensity:

Detection confidence drops significantly.

Classical DCP partially restores detection performance.

Hybrid residual refinement improves detection robustness.

Hybrid consistently shows confidence gains over classical.

Improvement increases under severe haze conditions.

The framework demonstrates that hybrid dehazing enhances both perceptual quality and detection reliability.

Installation
1. Clone the Repository
git clone https://github.com/22BCE0585/Hybrid-Dehazing-Detection.git
cd Hybrid-Dehazing-Detection
2. Create Virtual Environment (Recommended)

Windows

python -m venv venv
venv\Scripts\activate

Linux / Mac

python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

Running the Project
Multi-Image Detection Evaluation

Run:

python main.py

This performs:

Synthetic haze generation

Classical Dark Channel Prior restoration

Hybrid residual CNN refinement

YOLO-based object detection evaluation

Statistical confidence analysis

The script processes multiple images and prints average detection confidence results.

Run Interactive Web Application
streamlit run app.py

The web interface allows:

Uploading an image

Adjusting haze intensity

Viewing classical vs hybrid comparison

Interactive before/after slider

Transmission map visualization

Detection confidence analysis

Optional PSNR / SSIM computation (if ground truth provided)

Downloading hybrid output

Evaluation Metrics
Image Quality Metrics

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

Detection Metrics

Average detection confidence

Hybrid vs Classical confidence gain

Robustness under varying haze intensity

Project Structure
Hybrid-Dehazing-Detection/
│
├── app.py                     # Streamlit demo interface
├── main.py                    # Multi-image evaluation pipeline
├── architecture.png           # System architecture diagram
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── classical/             # Dark Channel Prior implementation
│   ├── cnn_refiner/           # Residual CNN model
│   ├── synthetic/             # Haze simulation module
│   └── detection/             # YOLO detection wrapper
│
├── evaluation/                # Metrics computation
│
├── data/
│   └── raw/                   # Sample input images (optional)
│
└── results/                   # Generated outputs (ignored in Git)
Technical Highlights

Physics-informed atmospheric modeling

Residual learning for transmission correction

Lightweight encoder-decoder CNN

Spatial haze simulation

Detection robustness evaluation

Modular and reproducible architecture

Why Hybrid Works

Classical DCP assumes:

Global atmospheric light

Uniform haze distribution

Under severe or spatially varying haze, these assumptions break.

The hybrid model:

Learns residual correction

Preserves structural consistency

Maintains detector compatibility

Reduces over-enhancement artifacts

Limitations

Synthetic haze may not perfectly match real-world atmospheric conditions

Extremely severe haze can still degrade detection performance

CNN was trained primarily on synthetic data

Future Improvements

Training on real-world haze datasets

Domain adaptation techniques

End-to-end joint dehazing-detection training

Lightweight mobile deployment

Edge-device optimization

Real-time video integration

Author

Your Name
Final Year Project
Computer Science / Artificial Intelligence

License

This project is developed for academic and research purposes.
It may be adapted for educational or research use.
