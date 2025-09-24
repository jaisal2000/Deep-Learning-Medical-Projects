# Brain Tumor Classification using Deep Learning: A Comparative Study of CNNs and Vision Transformers

## Project Overview

This project is a foundational study into the architectural differences between Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for the task of medical image classification. While both architectures have achieved state-of-the-art results, their internal mechanisms are fundamentally different. This work seeks to move beyond a simple comparison of accuracy metrics to investigate how these differences translate to performance, interpretability, and reliability in a clinical context.

### Research Questions (Motivation)

This project serves as a preliminary investigation into the following questions:

1.  **How does the architectural paradigm (local convolutions vs. global self-attention) influence a model's ability to localize salient pathological features in brain MRI scans?** Is the superior theoretical receptive field of a ViT reflected in more precise and interpretable class activation maps?
2.  **Are CNNs and ViTs equally robust to the types of data variations common in clinical settings?** How do they perform under synthetic domain shifts like imaging noise and contrast variations, and do their failure modes differ?

To address these questions, we implement, train, and rigorously evaluate a fine-tuned ResNet-34 (CNN) and a Vision Transformer (ViT) on a public brain tumor MRI dataset.

---

## Dataset Used

This project utilizes the **Brain Tumor MRI Dataset** available on Kaggle.

- **Source:** [Kaggle Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total Images:** 7,023
- **Image Format:** JPEG
- **Classes:** 4 categories
    - `glioma`
    - `meningioma`
    - `notumor` (no tumor)
    - `pituitary`
- **Structure:** The data is pre-organized into `Training` and `Testing` directories, which are further subdivided by class, making it suitable for standard supervised learning tasks.

---

## Key Features

- **Two Architectures:**
    1.  **Baseline CNN:** A fine-tuned ResNet-34 model.
    2.  **Transformer:** A fine-tuned Vision Transformer (ViT-Base).
- **Rigorous Evaluation:** Performance is measured using accuracy, F1-score, and ROC-AUC.
- **Interpretability Analysis:** Grad-CAM visualizations are used to understand and compare *how* each model makes its predictions.
- **Robustness Testing:** Models are challenged with synthetically corrupted data (noise, brightness shifts) to assess their real-world reliability.
- **Reproducibility:** The project is structured in a Colab notebook with clear, sequential cells. Final trained models are saved.

---

## Final Results Summary

This table summarizes the key performance metrics for the two final models on the unseen test set.

| Metric                        | CNN (ResNet-34)            | Vision Transformer (ViT)     |
| ----------------------------- | -------------------------- | ---------------------------- |
| **Base Accuracy**             | 98.86%                     | **99.31%**                   |
| **Base F1-Score (Macro)**     | 0.988                      | **0.993**                    |
| **Accuracy (Gaussian Noise)** | 93.36% (↓ 5.5%)            | **99.39%** (Slight change)       |
| **Accuracy (Brightness Shift)** | **98.47%** (↓ 0.4%)          | 96.49% (↓ 2.8%)              |
| **Interpretability (Grad-CAM)** | Broad, diffuse activation  | **Highly localized & precise** |

### Key Insights:
- The **Vision Transformer** achieved slightly higher overall accuracy and demonstrated superior, highly-focused interpretability.
- The **ViT** was exceptionally **robust to noise**, a significant advantage for medical imaging.
- The **CNN** was more **robust to changes in brightness/contrast**, showcasing the invariance of its learned features.

---

## Visualizations

### 1. Model Learning Curves

![alt text](image.png)

![alt text](image-1.png)

### 2. Interpretability with Grad-CAM

The Grad-CAM heatmaps reveal a key difference in how the models "see" the tumors.

**CNN (ResNet-34): Broad, contextual focus.**
![alt text](image-2.png)

**ViT: Precise, localized attention on the tumor.**
![alt text](image-3.png)

---
## Future Work and Research Directions

The findings from this project open up several promising avenues for future research, transitioning from comparative analysis to hypothesis-driven experimentation.

#### 1. Hypothesis-Driven Robustness Analysis
- **Observation:** The ViT is robust to noise, while the CNN is robust to brightness shifts.
- **Future Work:** Design a controlled experiment to systematically test the hypothesis that self-attention mechanisms are inherently better at ignoring unstructured, high-frequency noise, while convolutional filters are better at learning features invariant to global low-frequency changes. This would involve titrating noise levels and frequency bands and observing the performance degradation curves for each architecture.

#### 2. Self-Supervised Pre-training with In-Domain Data
- **Limitation:** Both models were pre-trained on ImageNet, which has a vastly different data distribution than medical images.
- **Future Work:** Implement a self-supervised pre-training strategy, such as a **Masked Autoencoder (MAE)**, on a large, unlabeled corpus of brain MRIs (e.g., from the TCGA database). The central research question would be: *Does in-domain self-supervised pre-training lead to more robust, data-efficient, and generalizable models compared to standard ImageNet pre-training?*

#### 3. Exploration of Hybrid Architectures
- **Observation:** CNNs and ViTs have complementary robustness profiles.
- **Future Work:** Design and evaluate a **hybrid CNN-Transformer architecture**. This could involve using a convolutional stem for patch embedding to leverage its stability with contrast changes, while retaining the Transformer body for its superior localization and noise robustness. The goal would be to create a model that inherits the best properties of both paradigms.

#### 4. Advanced Uncertainty Quantification and Out-of-Distribution Detection
- **Limitation:** MC Dropout provides a basic uncertainty estimate.
- **Future Work:** Implement and compare more advanced uncertainty quantification techniques like **Deep Ensembles** or **Conformal Prediction**. A key goal would be to build a system that can not only predict a class but also provide a calibrated confidence score, allowing it to reliably **reject** making predictions on out-of-distribution or ambiguous samples, which is critical for clinical safety.

#### 5. Extension to Multi-Modal and 3D Data
- **Limitation:** The current project uses 2D JPEG slices from a single MRI sequence.
- **Future Work:** Extend the models to handle full 3D volumetric data and multi-modal MRI sequences (T1, T2, FLAIR). This introduces the research challenge of **information fusion**: investigating how to best combine data from multiple sources within the model architecture, for example, through early fusion, late fusion, or dedicated cross-attention mechanisms between modalities.
---
## How to Run This Project

1.  **Setup (Cell 0):** Mount Google Drive and install required libraries like `timm` and `pytorch-grad-cam`.
2.  **Dataset (Cell 1):** Add your `kaggle.json` API key to Google Drive to download the dataset automatically.
3.  **Preprocessing (Cells 2-3):** Run the cells to prepare the data and create PyTorch DataLoaders.
4.  **Training (Cells 4-5):** Run the training scripts for the baseline CNN and the ViT. This will generate the `.pth` model files.
5.  **Analysis (Cells 5a, 5b, 7, 8):** Run the subsequent cells to generate performance plots, Grad-CAM visualizations, and robustness reports.
6.  **Saving & Inference (Cell 10):** This cell saves the final models to your Google Drive and contains an optional interactive demo to test with your own images.