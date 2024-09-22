# Water Segmentation Project

This project involves the segmentation of water bodies from multispectral satellite imagery using deep learning techniques. The models are implemented using different architectures and versions, leveraging both TensorFlow/Keras and PyTorch. The goal is to accurately identify and segment water bodies from satellite images with multiple spectral channels.

## Project Overview

The objective of this project is to segment water bodies from multispectral satellite images. The dataset consists of 12-channel images and their corresponding binary mask labels. Several versions of the segmentation model were implemented, utilizing U-Net and DeepLabv3 architectures with different variations and preprocessing techniques. This project demonstrates the model development process, optimization attempts, and the deployment of the best-performing model using Flask.

## Key Components

- **Multispectral Images:** The dataset includes 12 spectral channels, with each image being 128x128 pixels.
- **Binary Labels:** Corresponding binary mask labels, where `1` indicates water and `0` indicates non-water.
- **Models:** U-Net and DeepLabv3 architectures were implemented in TensorFlow/Keras and PyTorch. Various techniques were applied to improve the model performance, such as different normalizations, loss functions, and backbone networks.
- **Deployment:** The final model was deployed using Flask for water segmentation visualization.

---

## Notebooks Summary

### 1. **`Water_Segmentation.ipynb`**
   - **Description:** Initial version using U-Net architecture implemented from scratch for water segmentation.
   - **Preprocessing:** Custom normalization function for `.tif` images (12 channels).
   - **Architecture:** U-Net with convolutional and transpose convolution layers, sigmoid activation, and skip connections.
   - **Optimization:** Applied early stopping, model checkpoint, and learning rate reduction.
   - **Model:**  The model was saved as `u-net.keras`
   - **Results:**
     - Loss: `1.8003`
     - Accuracy: `77.22%`
     - Validation Accuracy: `58.78%`
   - **Notes:** The model was not highly effective, likely falling into a local minimum with moderate accuracy.

### 2. **`MinMax.ipynb`**
   - **Description:** Modified version with normalization using `image / 65535.0` to normalize all 12 channels.
   - **Architecture:** U-Net architecture with adjustments in learning rate and loss function (using dice loss and binary cross-entropy).
   - **Model:**  The model was saved as `minmax.keras`
   - **Results:**
     - Loss: `0.3470`
     - Accuracy: `91.20%`
     - Validation Accuracy: `92.32%`
   - **Notes:** Significant improvement over the first version, achieving over 91% accuracy.

### 3. **`DeepLabv3.ipynb`**
   - **Description:** This version uses DeepLabv3 with ResNet50 as the backbone, modified to accept 12 channels by adding custom Conv2D layers.
   - **Architecture:** ResNet50 with additional convolution layers for handling 12-channel input, using binary cross-entropy for training.
   - **Model:**  The model was saved as `deeplabv3.keras`
   - **Results:**
     - Loss: `0.3001`
     - Accuracy: `88.79%`
     - Validation Accuracy: `74.35%`
   - **Notes:** The model showed decent accuracy but suffered from overfitting.

### 4. **`9Channels.ipynb`**
   - **Description:** Best-performing version where the first 3 RGB channels were removed, leaving 9 channels for input. DeepLabv3 was implemented, and the model was adjusted to accept 9 channels.
   - **Preprocessing:** MinMax normalization on 9 channels (excluding RGB).
   - **Model:**   The model was saved as `9channels.keras`.
   - **Results:**
     - Loss: `0.1992`
     - Accuracy: `91.53%`
     - Validation Accuracy: `90.41%`
   - **Notes:** This version yielded the best results, with over 91% accuracy.

### 5. **`torch.ipynb`**
   - **Description:** PyTorch implementation of the entire project, the same sturcture as `9Channels.ipynb`
   - **Preprocessing:** Same dataset and preprocessing as earlier versions, but implemented with PyTorch.
   - **Model:** The model was saved as `torch_9channels.pth`.
   - **Results:**
     - Loss: `0.0424`
     - Accuracy: `73.29%`
     - Validation Accuracy: `77.79%`
   - **Notes:** This version did not perform well, potentially due to overfitting. Future parameter adjustments are needed.

---

## Deployment

The best-performing model (`9channels.keras`) was deployed using Flask. The Flask app allows users to visualize the original image in RGB, NDWI (Normalized Difference Water Index) composite, and the model’s prediction in grayscale and binary mask format.

- **Files included in deployment:**
  - `app.py`: Flask application.
  - `templates/`: Folder containing HTML templates for visualization.
  - `requirements.txt`: Dependencies for the Flask app.

---

## Conclusion

This project showcases multiple versions of water segmentation models with varying architectures and preprocessing techniques. The best-performing model achieved an accuracy of over 91%, and it has been deployed for real-time predictions using a Flask web application.
