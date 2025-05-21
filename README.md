# Brain Tumor Detection using Deep Learning

## Overview
This Jupyter notebook implements a deep learning-based approach to detect brain tumors from MRI images. The model is designed to classify MRI scans into two categories: healthy (no tumor) and tumor (presence of tumor). The notebook covers the entire pipeline from data loading and preprocessing to model training and evaluation.

## Dataset
The dataset used in this project is from Kaggle:
- **Source**: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Contents**:
  - 85 healthy brain MRI images (no tumor)
  - 86 brain MRI images with tumors
- **Image Specifications**: All images are resized to 128x128 pixels and converted to RGB format

## Notebook Structure
1. **Data Loading and Preprocessing**
   - Mounts Google Drive to access dataset
   - Loads and resizes images
   - Separates images into healthy and tumor categories
   - Visualizes sample images from both classes

2. **Data Preparation**
   - Combines healthy and tumor images into a single dataset
   - Performs train-test split (implied though not explicitly shown)
   - Applies necessary transformations (resizing, normalization)

3. **Model Architecture**
   - Uses PyTorch for deep learning implementation
   - Includes convolutional neural network (CNN) layers
   - Implements data augmentation techniques

4. **Training and Evaluation**
   - Defines loss function and optimizer
   - Includes training loop
   - Evaluates model performance using metrics like:
     - Confusion matrix
     - Accuracy score

5. **Visualization**
   - Plots sample images from both classes
   - Visualizes training progress and results

## Dependencies
The notebook requires the following Python libraries:
- NumPy
- PyTorch
- OpenCV (cv2)
- Matplotlib
- scikit-learn
- Google Drive (for data access when running in Google Colab)

## Usage
1. Upload the dataset to your Google Drive
2. Open the notebook in Google Colab or Jupyter Notebook
3. Mount your Google Drive when prompted
4. Update the dataset paths if necessary:
   ```python
   tumor_path = "/content/drive/MyDrive/brain_tumor_dataset/yes/*.jpg"
   healthy_path = "/content/drive/MyDrive/brain_tumor_dataset/no/*.jpg"