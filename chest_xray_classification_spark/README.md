# Chest X-Ray Classification with PySpark

This project demonstrates a distributed image classification pipeline using **Apache Spark** and **PySpark**.

The goal is to automatically classify chest X-ray scans into two categories: **NORMAL** and **PNEUMONIA**, based on a small dataset of medical images.

## 🔍 Overview

We implement the following steps:

1. **Image loading and vectorization**: convert X-ray images into pixel vectors using a UDF in PySpark.
2. **Dimensionality reduction**: reduce feature size using PCA (Principal Component Analysis).
3. **Classification**: train a logistic regression model to predict the class of unseen X-rays.
4. **Evaluation**: compute accuracy on a test set and inspect predictions.
5. **DAG analysis**: use `.explain(True)` and `.repartition()` to understand and control Spark's execution flow.

## 📊 Technologies

- Apache Spark / PySpark
- Python 3
- Spark MLlib (PCA, LogisticRegression)
- Dataset: Chest X-Ray Images (subset)

## 📁 Dataset Setup

You can use a small subset of the **Chest X-Ray Images** dataset (e.g., from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)).

**Directory structure expected by the notebook:**

./data/chest_xray/train/
├── NORMAL/
│ ├── image1.jpeg
│ ├── image2.jpeg
│ └── ...
└── PNEUMONIA/
├── image1.jpeg
├── image2.jpeg
└── …


- You only need a small number of images for this POC (~10–15 per class).
- Images should be JPEG or PNG format.

Update the notebook path if needed to match your local structure.

## ✅ Results

- Model trained using 5 principal components
- Accuracy on test set: **100%** (on a small balanced sample)
- Includes analysis of the Spark DAG and partitioning behavior

## 🚀 Getting Started

To run this notebook:

1. Clone this repository
2. Make sure you have Spark and Python installed (or use Databricks)
3. Place the dataset in the expected folder (`./data/chest_xray/train/`)
4. Open the notebook and follow the steps

## 📌 Notes

This is a simplified proof-of-concept (POC). For production use:
- Larger datasets are required
- Deep learning models (e.g., CNNs) are more suitable
- Preprocessing and augmentation should be improved

## 📬 Contact

Feel free to reach out if you have questions or feedback!
