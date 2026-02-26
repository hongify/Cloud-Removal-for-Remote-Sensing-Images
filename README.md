# Multi-Modal Cloud Removal via Multi-Branch cGAN

### Project Overview
This project implements an advanced cloud removal system for remote sensing imagery. By leveraging Multi-Modal Data Fusion (Landsat-8, Sentinel-1 SAR, and MODIS), the model reconstructs high-resolution optical channels even under heavy cloud cover. The core architecture is based on a Conditional Generative Adversarial Network (cGAN), optimized for high-speed inference and structural integrity.

---

### Model Architecture
The system utilizes a specialized Multi-Branch Generator and a PatchGAN Discriminator to balance spectral accuracy with spatial texture.

* **Multi-Branch Encoder**: 
    * Landsat Branch: Processes 6 optical bands + 1 Cloud Mask (7-channel input).
    * MODIS Branch: Extracts global optical priors from 6 MODIS bands.
    * Sentinel-1 Branch: Captures ground structural information through cloud-penetrating SAR (VV, VH).
* **Bottleneck**: Features 6 Residual Blocks to maintain deep feature consistency and prevent vanishing gradients.
* **Decoder**: Reconstructs 6-channel outputs (RGB, NIR, SWIR1, SWIR2) using Transposed Convolution and Sigmoid activation.
* **Discriminator**: A 70x70 PatchGAN that penalizes blurry results, forcing the generator to produce sharp edges and realistic textures.

---

### Key Technical Features
* **Soft-Mask Integration**: Instead of binary masks, the model utilizes soft masks (0, 0.5, 1) to account for cloud opacity, allowing for fine-grained reconstruction of thin clouds.
* **Composite Loss Function**:
    * Weighted L1 Loss: Prioritizes reconstruction error in cloudy regions.
    * SSIM Loss: Ensures structural and perceived quality similarity.
    * Edge Loss: Sharpens boundaries using Sobel-filter based gradient matching.
* **Selective Restoration**: During inference, only the cloudy regions (defined by the mask) are replaced by generated pixels, while clear regions retain 100% of the original sensor data.

---

### Environment and Requirements
* OS: Ubuntu 20.04+ (Recommended)
* Language: Python 3.x
* Framework: PyTorch (with CUDA support)
* Libraries: NumPy, Rasterio, Matplotlib, tqdm

---

### Usage Instructions

#### 1. Data Preparation
Construct a 15-channel tensor dataset following this order:
1. Channels 0-5: Landsat-8 (RGB, NIR, SWIR1, SWIR2)
2. Channels 6-11: MODIS bands
3. Channels 12-13: Sentinel-1 SAR (VV, VH)
4. Channel 14: Cloud Mask (0 for clear, 0.5 for thin, 1 for thick)

#### 2. Training
Execute the training script with desired hyperparameters:
```bash
python train.py --batch_size 16 --epochs 100 --lr 0.0002