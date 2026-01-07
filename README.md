# Alzheimer’s Disease Binary Classification (ND vs. VMD)

![Project Status](https://img.shields.io/badge/Status-Completed-success) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)

## 📌 Project Overview
This project focuses on the binary classification of Alzheimer's Disease using MRI scans, specifically distinguishing between **Non-Demented (ND)** and **Very Mild Demented (VMD)** patients.

The goal is to assist in early diagnosis using Deep Learning, prioritizing the minimization of **False Negatives** (missed diagnoses), which is the most critical error in medical contexts.

* **Context:** Master in Data Engineering and Data Science @ **UTAD (Portugal)**.
* **Author:** Carlos Roberto Souza Garcia Filho.

## 🧠 The Challenge
Detecting Alzheimer's at the "Very Mild" stage is visually challenging due to the subtle differences in brain tissue contrast.
This project moves beyond standard accuracy metrics, conducting a comparative study of 4 approaches to find the architecture that offers the best **Recall (Sensitivity)** without overfitting.

**Dataset:** [Alzheimer's Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/kanaadlimaye/alzheimers-classification-dataset)
* **Input:** MRI Images (resized to 224x224).
* **Classes:** Non-Demented (Healthy) vs. Very Mild Demented.

## 🧪 Methodology & Experiments
I implemented and compared four distinct experimental setups:

1.  **Baseline CNN:** A simple 2-block Convolutional Neural Network to establish a performance benchmark.
2.  **Data Augmentation:** Applying rotation, zoom, and shifts to test if artificially increasing dataset diversity aids generalization.
3.  **Deep CNN + BatchNormalization + Dropout (Proposed Solution):**
    * **Increased Depth:** 4 Convolutional blocks (up to 256 filters) to capture high-level abstract features.
    * **Batch Normalization:** Applied after every convolution to stabilize learning and allow faster convergence.
    * **Dropout (0.5):** Robust regularization applied to the dense layers to prevent overfitting.
    * **Fine-Tuning:** Lower learning rate (`1e-4`) for precise convergence.
4.  **Transfer Learning (VGG16):** Using a pre-trained VGG16 model (frozen ImageNet weights) with a custom classifier head.

## 📊 Performance Analysis

Contrary to the common belief that Transfer Learning (VGG16) is always superior, this study revealed that **simpler, custom-built architectures** performed significantly better for this specific grayscale medical task.

### 1. Quantitative Results

| Model | Accuracy | Recall (VMD) | Key Observation |
| :--- | :--- | :--- | :--- |
| **1. Baseline** | 97% | 0.94 | High accuracy, but prone to overfitting (unstable validation loss). |
| **2. Data Augmentation** | 68% | 0.48 | **Underfitting.** Transformations were too aggressive for the dataset size. |
| **3. Deep CNN + BN + Dropout** | **98%** | **0.97** | **Best Model.** High stability, fast convergence, and excellent generalization. |
| **4. Transfer Learning (VGG16)** | 84% | 0.86 | **Underfitting.** Generic ImageNet weights failed to capture subtle MRI textures. |

![Comparison Charts](images/performance_comparison.png)

### 2. Confusion Matrix Analysis (The "Tie-Breaker")
While the Baseline and the Deep Model achieved similar numeric scores, the detailed analysis reveals why Model 3 is superior for production.

![Confusion Matrix Grid](images/confusion_matrix_grid.png)

* **VGG16 Failure:** Missed a significant number of positive cases (High False Negatives).
* **The Winner (Model 3):** The combination of **4 Convolutional Layers** with **Batch Normalization** allowed the model to learn more complex features than the Baseline, while **Dropout** kept the training stable. It achieved the lowest False Negative rate, missing only 7 cases in the entire test set.

## 🏆 Why Model 3 Won?
**1. Capacity vs. Control:**
The Baseline (2 layers) was accurate but unstable. By increasing the depth to **4 layers (256 filters)**, we gave the model "brain power" to understand subtle MRI textures. Normally, this would cause overfitting, but **Batch Normalization** and **Dropout** acted as "guardrails", keeping the model generalized.

**2. Domain Specificity:**
The VGG16 model, trained on colorful real-world images (ImageNet), struggled to adapt its frozen weights to grayscale medical scans. The Custom CNN learned these specific features from scratch, proving that **targeted architecture design beats generic pre-trained models.**

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Metrics:** Scikit-learn (Confusion Matrix, Classification Report)
* **Environment:** Google Colab (GPU T4)

## 📂 Repository Structure
* `notebooks/`: Contains the Jupyter Notebook with the complete training pipeline.
* `images/`: Confusion matrices, loss curves, and analysis charts.

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/alzheimer-detection-cnn.git](https://github.com/YOUR_USERNAME/alzheimer-detection-cnn.git)
    ```
2.  Install dependencies:
    ```bash
    pip install tensorflow pandas matplotlib seaborn scikit-learn kagglehub
    ```
3.  Run the notebook. The dataset is automatically downloaded via the `kagglehub` API (no manual setup required).

---
*Author: Carlos Roberto Souza Garcia Filho*
