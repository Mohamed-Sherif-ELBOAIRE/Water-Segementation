# Water Segmentation Project

This project involves the segmentation of water bodies from multispectral satellite imagery using deep learning techniques. The model is implemented using TensorFlow/Keras, leveraging the U-Net architecture, which is widely used for image segmentation tasks.

## Project Overview

This project aims to accurately identify and segment water bodies from satellite images with multiple spectral channels. The dataset consists of 12-channel images and corresponding binary mask labels. The model used for segmentation is based on the U-Net architecture, optimized with various data preprocessing techniques and evaluated using standard image segmentation metrics.

### Key Components

- **Multispectral Images**: The dataset includes 12 spectral channels, where each image is of size 128x128 pixels.
- **Binary Labels**: Corresponding labels are provided as binary masks (1 for water and 0 for non-water).
- **Model**: A U-Net architecture is used for segmentation, which is a convolutional neural network (CNN) designed for biomedical image segmentation tasks but applied here for water segmentation.
