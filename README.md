# IDM-VTON: AI-Powered Virtual Try-On System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)
![License](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

_A high-fidelity AI-powered virtual clothing try-on system using advanced diffusion models and computer vision techniques._

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Model Details](#model-details) • [Contributing](#contributing)

</div>

---

## 📋 Overview

**IDM-VTON** is a cutting-edge virtual try-on system that leverages deep learning and diffusion models to realistically simulate clothing on human models. By combining Stable Diffusion XL, human pose estimation, clothing segmentation, and image inpainting techniques, IDM-VTON generates photorealistic try-on results that maintain the structural integrity and fit of garments.

This project provides both **web-based** (Gradio) and **desktop** (Tkinter) interfaces for seamless integration into e-commerce platforms and fashion applications.

---

## ✨ Key Features

- **🎯 High-Fidelity Results**: Photorealistic clothing transfer with precise garment fitting
- **🤖 Advanced AI Models**:
  - Stable Diffusion XL with custom inpainting capabilities
  - DensePose for detailed human body structure
  - OpenPose for robust joint detection
  - Human Parsing for accurate clothing segmentation
- **🌐 Multiple Interfaces**:
  - Web UI powered by Gradio (easy cloud deployment)
  - Desktop application with Tkinter GUI
  - REST API for programmatic access
- **⚡ GPU Optimized**: Efficient inference with half-precision (FP16) support
- **🎨 Customizable**: Support for various garment types and descriptions
- **📦 Production Ready**: Complete error handling and retry mechanisms

---

## 🏗️ Project Architecture

```
IDM-VTON/
├── app.py                          # Gradio web interface
├── api.py                          # Tkinter desktop application
├── apply_net.py                    # Core inference logic
├── utils_mask.py                   # Mask processing utilities
│
├── src/                            # Core pipeline modules
│   ├── tryon_pipeline.py           # Custom SDXL inpainting pipeline
│   ├── unet_hacked_tryon.py        # Modified UNet for try-on
│   ├── unet_hacked_garmnet.py      # Garment encoding network
│   ├── attentionhacked_*.py        # Custom attention mechanisms
│   ├── transformerhacked_*.py      # Modified transformer layers
│   └── unet_block_hacked_*.py      # Customized UNet blocks
│
├── preprocess/                     # Preprocessing modules
│   ├── humanparsing/               # Clothing segmentation
│   └── openpose/                   # Pose estimation
│
├── ip_adapter/                     # Image conditioning adapter
│   ├── ip_adapter.py               # Main IP-Adapter implementation
│   ├── ip_adapter_faceid.py        # Face ID preservation
│   ├── resampler.py                # Cross-attention resampler
│   └── custom_pipelines.py         # Custom diffusion pipelines
│
├── densepose/                      # DensePose implementation
├── detectron2/                     # Detectron2 framework
├── configs/                        # Model configurations
├── ckpt/                          # Checkpoint storage
│   ├── densepose/                 # DensePose weights
│   ├── humanparsing/              # Parsing model weights (ONNX)
│   └── openpose/                  # OpenPose weights
│
└── example/                        # Example images for demo
    ├── cloth/                      # Sample garment images
    └── human/                      # Sample person images
```

---

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **CUDA 11.8+** (for GPU support)
- **Git LFS** (for large model files)
- Minimum **8GB VRAM** (16GB+ recommended for better performance)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/IDM-VTON.git
cd IDM-VTON
git lfs install
git lfs pull
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package         | Version | Purpose                         |
| --------------- | ------- | ------------------------------- |
| `torch`         | 2.0.1   | Deep learning framework         |
| `torchvision`   | 0.15.2  | Computer vision utilities       |
| `diffusers`     | 0.25.0  | Diffusion model implementations |
| `transformers`  | 4.36.2  | Pre-trained transformer models  |
| `opencv-python` | 4.7.0   | Image processing                |
| `onnxruntime`   | 1.16.2  | ONNX model inference            |
| `gradio`        | 4.24.0  | Web interface framework         |

---

## 🚀 Usage

### Option 1: Web Interface (Gradio)

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:7860`

**Features:**

- Upload person image and garment image
- Add custom garment description
- Adjust denoising steps and seed for fine-tuning
- Real-time processing with visual feedback

### Option 2: Desktop Application (Tkinter)

```bash
python api.py
```

**Features:**

- Browse local files easily
- Interactive GUI with image preview
- Save output directly to disk
- Retry mechanism for failed requests

### Option 3: Programmatic API

```python
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
from transformers import AutoTokenizer, CLIPVisionModelWithProjection

# Initialize models
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "yisol/IDM-VTON",
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Run inference
result = pipe(
    prompt="A person wearing a red shirt",
    image=person_image,
    mask_image=clothing_mask,
    num_inference_steps=30,
    guidance_scale=7.5
)

output = result.images[0]
output.save("try_on_result.png")
```

---

## 🧠 Model Details

### Core Components

#### 1. **Stable Diffusion XL (SDXL)**

- Base diffusion model for high-quality image generation
- Custom inpainting pipeline for clothing replacement
- 1024×1024 resolution support
- Two-stage text encoding (CLIP-L + CLIP-G)

#### 2. **IP-Adapter (Image Prompt Adapter)**

- Enables image-level conditioning without fine-tuning
- Resampler module for cross-attention injection
- Face ID preservation variant for identity consistency
- Lightweight and efficient

#### 3. **DensePose**

- Estimates dense human body surface correspondence
- Provides detailed body structure information
- Supports multiple backbones (R50, R101)
- Essential for body-aware inpainting

#### 4. **OpenPose**

- Detects 25 body joints and pose landmarks
- Provides coarse body structure as auxiliary information
- Lightweight and efficient
- Used for pose-aware adjustments

#### 5. **Human Parsing**

- Segments clothing regions from person images
- Identifies garment types (shirt, pants, dress, etc.)
- ONNX-optimized for fast inference
- Supports ~18 clothing categories

### Training Configuration

| Parameter         | Value                     |
| ----------------- | ------------------------- |
| Base Architecture | Stable Diffusion XL       |
| Input Resolution  | 1024×1024                 |
| Guidance Scale    | 7.5                       |
| Default Steps     | 30                        |
| Batch Size        | 1 (configurable)          |
| Inference Backend | Diffusers with Accelerate |
| Precision         | FP16 (half-precision)     |

---

## 📊 Performance & Benchmarks

| Metric                        | Value                         |
| ----------------------------- | ----------------------------- |
| Inference Time (single image) | ~45-60 seconds (GPU)          |
| Memory Usage                  | ~6-8 GB VRAM                  |
| Image Quality                 | High-fidelity, photorealistic |
| Supported Resolutions         | Up to 1024×1024               |
| Batch Processing              | Yes (with memory constraints) |

---

## 🔧 Configuration

All configurations are managed through YAML files in the `configs/` directory:

```yaml
# Example configuration structure
model:
  base_path: "yisol/IDM-VTON"
  device: "cuda"
  precision: "float16"

inference:
  num_steps: 30
  guidance_scale: 7.5
  seed: 42

preprocessing:
  densepose_enabled: true
  humanparsing_enabled: true
  openpose_enabled: true
```

---

## 📝 Input/Output Specifications

### Input Requirements

**Person Image:**

- Format: PNG, JPG, JPEG
- Resolution: Recommended 512-1024px
- Content: Single person in full body view
- Quality: Minimum 72 DPI

**Garment Image:**

- Format: PNG, JPG, JPEG
- Background: White or transparent preferred
- Resolution: Minimum 256×256px
- Content: Clear garment image

**Garment Description:**

- Natural language description (e.g., "Red silk shirt with buttons")
- Length: 10-100 characters
- Language: English

### Output Specifications

**Result Image:**

- Format: PNG
- Resolution: 1024×1024
- Format: RGBA (with transparency support)
- File Size: ~500KB-2MB

---

## 🎯 Use Cases

1. **E-Commerce**: Virtual try-on for online fashion retailers
2. **Fashion Design**: Design iteration and visualization
3. **Social Media**: Interactive fashion content creation
4. **Personal Styling**: Try before you buy applications
5. **Virtual Fashion Shows**: Digital clothing showcases

---

## ⚠️ Limitations

- Works best with clear, well-lit person and garment images
- May struggle with complex garment designs or unusual poses
- Requires GPU for reasonable inference speeds
- Cannot modify garment shape dramatically (only color/texture)
- Best results with conventional garment types

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black src/ app.py api.py
```
