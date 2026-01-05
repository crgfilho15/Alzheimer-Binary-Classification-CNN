# Alzheimer’s Disease Binary Classification (ND vs. VMD) using CNNs

## 📌 Project Overview
This project focuses on the binary classification of Alzheimer's Disease using MRI scans, specifically distinguishing between **Non-Demented (ND)** and **Very Mild Demented (VMD)** patients. The goal is to assist in early diagnosis using Deep Learning.

Developed as part of the **Master in Data Engineering and Data Science** at **UTAD (Portugal)**.

## 🧠 The Challenge
Early detection of Alzheimer's is critical but challenging. This project compares four different Deep Learning approaches to find the best balance between model complexity and generalization, specifically targeting the reduction of False Negatives (missing a diagnosis).

**Dataset:** [Alzheimer's Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/kanaadlimaye/alzheimers-classification-dataset)
* **Input:** MRI Images (resized to 224x224)
* **Classes:** Non-Demented (Healthy) vs. Very Mild Demented

## 🧪 Experiments & Methodology
I conducted four distinct experiments to optimize performance and analyze model behavior:

1.  **Baseline CNN:** A simple 2-block CNN to establish a performance benchmark.
2.  **Data Augmentation:** Applying rotation, zoom, and shifts to combat overfitting.
3.  **Deep Architecture + Dropout:** Increasing model capacity while adding regularization (Dropout 0.5).
4.  **Transfer Learning (VGG16):** Using a pre-trained VGG16 model (feature extraction) with a custom classifier head.

## 📊 Key Results

| Model | Accuracy | Recall (ND) | Recall (VMD) | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline** | 98% | 0.98 | 0.97 | High accuracy but showed signs of overfitting early on. |
| **2. Data Augmentation** | 69% | 0.83 | 0.51 | **Underfitting.** Transformations were too aggressive for the dataset size. |
| **3. Deep CNN + Dropout** | **98%** | **0.97** | **0.99** | **Best Model.** Excellent generalization and lowest False Negative rate (2). |
| **4. Transfer Learning (VGG16)** | 77% | - | - | **Underfitting.** Frozen layers were not sufficient for specific feature extraction. |

### Why Model 3 Won?
While Transfer Learning (VGG16) is often the "go-to" solution, in this specific binary context, the frozen VGG16 layers were too generic. The **Custom Deep CNN with Dropout** (Model 3) managed to learn specific MRI features without memorizing the training data, achieving a nearly perfect confusion matrix.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
* **Environment:** Google Colab (GPU accelerated)

## 📂 Repository Structure
* `notebooks/`: Contains the Jupyter Notebook with the complete training pipeline.
* `reports/`: Full PDF report with theoretical analysis (in Portuguese).
* `images/`: Confusion matrices and loss curves.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install tensorflow pandas matplotlib seaborn kagglehub`
3. Run the notebook. The dataset is automatically downloaded via the `kagglehub` API.

---
*Author: Carlos Roberto Souza Garcia Filho*
