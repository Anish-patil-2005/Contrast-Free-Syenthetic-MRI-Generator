
---

# 🧠 Contrast-Free Synthetic MRI Generation

### Powered by Diffusion Models & MONAI

This project uses a **Diffusion Model (U-Net)** to generate synthetic **T1-Contrast (T1-Gd)** MRI images from non-contrast sequences (**T1, FLAIR, and BRAVO**). The goal is to produce medical-grade synthetic enhancement without the need for injecting gadolinium contrast agents.

* Live at :  https://huggingface.co/spaces/anishpatil/neuro-synth-ai

---

## 🛠️ Setup Instructions

### 1. Prerequisites

Ensure you have **Python 3.10+ (64-bit)** installed.

### 2. Installation

Clone the repository and install the dependencies listed in `requirements.txt`:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt

```

### 3. Data Structure

Place your BraTS-METS dataset in the following directory structure:

```text
data/
└── raw/
    ├── train/ (105+ patients)
    └── test/  (Independent test set)

```

---

## 🚀 How to Run

### **Training the Model**

To start the training process from scratch:

```bash
python train.py

```

*Note: On a CPU, this may take several hours. The model saves checkpoints to the `/models` folder.*

### **Testing/Inference**

To generate a synthetic image using the trained weights:

```bash
python test_model.py

```

This will pick a patient from the `test` folder, run the reverse diffusion process, and save a comparison image as `test_result.png`.

---

## 🧬 Project Architecture

* **Model:** DiffusionModelUNet (via MONAI)
* **Algorithm:** Denoising Diffusion Probabilistic Models (DDPM)
* **Preprocessing:** Min-Max Intensity Scaling & 2D Slicing
* **Framework:** PyTorch & MONAI

---

## 📊 Current Results

The model is currently trained for 10 epochs. While the pipeline is fully functional, additional training (50-100 epochs) on a GPU is recommended to move from the "noise" phase to high-fidelity brain synthesis.

---


