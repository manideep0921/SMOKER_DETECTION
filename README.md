# 🚬 Smoking Detection Using Deep Learning (CNN)

This project aims to **detect whether a person is a smoker or non-smoker** based on facial images using a pretrained deep learning model. The dataset was collected from **Kaggle**, processed, and used to train a **transfer learning-based CNN model (InceptionResNetV2)**.

---

## 📂 Project Structure

├── DATA SET # Original dataset (Kaggle)
├── original_images # Split dataset (train/val)
├── roi_dataset # Face cropped dataset
├── model_training.py # Model training script
├── README.md # Project documentation

yaml
Copy code

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|-------------|
| Framework | TensorFlow / Keras |
| Model | InceptionResNetV2 |
| Image Processing | OpenCV |
| Metrics | Scikit-learn |
| Language | Python |

---

## 📊 Workflow Overview

### 1️⃣ Dataset Split
- Original dataset was divided into **train (80%)** and **validation (20%)**
- Labels organized into:
  - `Smoker`
  - `Non-Smoker`

```python
split_ratio = 0.8
for split in ['train', 'val']:
    for class_name in ['Smoker', 'Non-Smoker']:
        os.makedirs(...)
