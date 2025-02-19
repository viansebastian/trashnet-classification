# 🗑️ TrashNet Classification

This repository contains a **TrashNet Classification** model, designed to classify waste into categories using machine learning. The model is built and trained using TensorFlow, and incorporates Focal Loss to address class imbalance.

## 🎯 Features

- **Trash Classification**: Accurately classify waste into predefined categories.
- **Focal Loss Optimization**: Enhanced performance on imbalanced datasets.
- **Ready to Use**: Easily integrate the model into any Python-based pipeline.

---

## 🚀 Quick Start Guide

Follow the steps below to get started with the project.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/viansebastian/trashnet-classification
cd trashnet-classification
```

### 2️⃣ Install Dependencies
Ensure that Python is installed on your system. Then, install the required libraries:
```bash
pip install -r requirements.txt
```

### 3️⃣ Load and Use the Model
Load the trained model and use it for predictions:

```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
focal_model = load_model("focal_model.keras")

# Load an image for prediction (example)
image = np.array(Image.open("sample_image.jpg").resize((224, 224))).astype("float32") / 255.0
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Predict the class
predicted_class = focal_model.predict(image)
print(f"Predicted Class: {np.argmax(predicted_class)}")
```

---

## 🧾 Requirements

- Python >= 3.10
- TensorFlow >= 2.9
- NumPy

---

Let me know if you'd like to add more details or make adjustments!