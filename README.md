# Conditional GAN for Skin Cancer Image Generation 🏥🔬

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0+-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art implementation of **Conditional Generative Adversarial Networks (cGAN)** for generating synthetic skin cancer images across 9 different classes using deep learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

This project implements a **Conditional GAN** using TensorFlow/Keras to generate realistic synthetic medical images for skin cancer classification. The model learns to generate class-specific images conditioned on disease labels, which can be valuable for:

- **Medical Training**: Creating diverse training datasets
- **Data Augmentation**: Balancing imbalanced medical datasets
- **Research**: Studying disease characteristics and patterns
- **Privacy**: Generating synthetic data without exposing real patient information

## ✨ Features

### Core Features
- ✅ **Dynamic Image Loading**: Reads training images directly from directory structure
- ✅ **Multi-Class Support**: Generates images for 9 different skin cancer classes
- ✅ **Conditional Generation**: Class-specific image synthesis
- ✅ **GPU Acceleration**: Optimized for NVIDIA GPUs (tested on P100)
- ✅ **Visualization**: Progress tracking during training
- ✅ **Model Persistence**: Save and load trained models

### Advanced Techniques
- 🚀 LeakyReLU activations for stable training
- 🎯 Batch normalization for faster convergence
- 🔄 Label conditioning in both Generator and Discriminator
- 💾 Automatic checkpoint saving

## 📊 Dataset

**Dataset Used**: [Skin Cancer: Malignant vs. Benign (ISIC)](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

### Classes Included:
1. Actinic Keratosis
2. Basal Cell Carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented Benign Keratosis
7. Seborrheic Keratosis
8. Squamous Cell Carcinoma
9. Vascular Lesion

### Data Specifications:
- **Image Size**: 64×64 pixels (RGB)
- **Format**: JPEG
- **Organization**: Class-based directory structure
- **Preprocessing**: Normalization to [-1, 1] range

## 🏗️ Architecture

### Generator Network
```
Input: [Noise Vector (100D) + Class Label]
    ↓
Dense(8×8×256) + BatchNorm + LeakyReLU
    ↓
Reshape(8, 8, 256)
    ↓
Conv2DTranspose(128) + BatchNorm + LeakyReLU  [16×16×128]
    ↓
Conv2DTranspose(64) + BatchNorm + LeakyReLU   [32×32×64]
    ↓
Conv2DTranspose(3) + Tanh                     [64×64×3]
    ↓
Output: Generated Image (64×64×3)
```

### Discriminator Network
```
Input: [Image (64×64×3) + Class Label]
    ↓
Conv2D(64) + LeakyReLU                        [32×32×64]
    ↓
Conv2D(128) + BatchNorm + LeakyReLU           [16×16×128]
    ↓
Conv2D(256) + BatchNorm + LeakyReLU           [8×8×256]
    ↓
Flatten + Dense(1) + Sigmoid
    ↓
Output: Real/Fake Classification
```

### Clone Repository
```bash
git clone https://github.com/abdulhasibuddin/cgan-for-medical-images.git
cd cgan-for-medical-images
```

## 💻 Usage

### Basic Training

```python
# Import necessary libraries
import numpy as np
from tensorflow import keras
from PIL import Image

# Configure paths
training_dir = '/path/to/skin-cancer-dataset/Train/'
output_dir = '/path/to/output/generated_images/'

# Load and preprocess data
# (See notebook for complete implementation)

# Train the model
gan.fit(dataset, epochs=100000, verbose=1)
```

### Generate Images

```python
# Generate images for specific class
noise = np.random.normal(0, 1, (16, latent_dim))
labels = np.array([class_id] * 16)  # 0-8 for different classes

generated_images = generator.predict([noise, labels])

# Save generated images
for i, img in enumerate(generated_images):
    img = (img + 1) / 2.0  # Denormalize
    Image.fromarray((img * 255).astype(np.uint8)).save(f'generated_{i}.png')
```

### Load Pretrained Models

```python
# Load saved models
generator = keras.models.load_model('generator.keras')
discriminator = keras.models.load_model('discriminator.keras')
```

## 📈 Results

### Training Metrics
- **Training Epochs**: 100000
- **Batch Size**: 16
- **Latent Dimension**: 100
- **GPU**: NVIDIA Tesla P100
- **Training Time**: ~8 hours 45 minutes

### Sample Generated Images

The model successfully generates class-specific synthetic skin cancer images with realistic textures and patterns. Generated images are saved at regular intervals during training.

### Model Performance
- Generator Loss: Stable
- Discriminator Loss: Maintains balance with generator
- Image Quality: Optimal-quality, diverse samples per class

## 🔧 Technical Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `latent_dim` | 100 | Noise vector dimension |
| `img_shape` | (64, 64, 3) | Input image dimensions |
| `num_classes` | 9 | Number of disease classes |
| `batch_size` | 16 | Training batch size |
| `epochs` | 100000 | Total training epochs |
| `learning_rate` | 0.0002 | Adam optimizer LR |
| `beta_1` | 0.5 | Adam momentum term |

### Key Techniques

1. **Label Conditioning**: Both Generator and Discriminator receive class labels as additional inputs
2. **LeakyReLU**: Alpha=0.2 for preventing dead neurons
3. **Batch Normalization**: Applied after most layers for stability
4. **Binary Cross-Entropy Loss**: Standard GAN loss function
5. **Adam Optimizer**: Learning rate 0.0002, beta_1=0.5


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- [ ] Implement Progressive GAN architecture
- [ ] Add Self-Attention mechanisms
- [ ] Integrate Spectral Normalization
- [ ] Support for higher resolution images (128×128, 256×256)
- [ ] Add FID (Fréchet Inception Distance) evaluation
- [ ] Web interface for interactive generation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: [ISIC - International Skin Imaging Collaboration](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
- **Inspiration**: Original cGAN paper by Mehdi Mirza and Simon Osindero
- **Framework**: TensorFlow/Keras team for excellent deep learning tools
- **Platform**: Kaggle for providing free GPU resources

## 📧 Contact

For questions or collaboration opportunities:

- **GitHub**: [@abdulhasibuddin](https://github.com/abdulhasibuddin)
- **Email**: abdulhasibuddin2@gmail.com
- **LinkedIn**: [Abdul Hasib Uddin](https://linkedin.com/in/abdul-hasib-uddin)

---

⭐ **Star this repository** if you find it helpful!

🐛 **Report bugs** by opening an issue

💡 **Suggest features** via pull requests

---

*Made with ❤️ for advancing medical AI research*
