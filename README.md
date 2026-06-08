# 🩺 Breast Cancer Classification using Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)](https://keras.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellowgreen?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> A Deep Learning project that uses a Neural Network built with **TensorFlow/Keras** to classify breast cancer tumors as **Malignant** or **Benign** using medical diagnostic data.

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early diagnosis plays a critical role in successful treatment and survival.

This project demonstrates how a **Neural Network** can be trained on medical data to predict whether a tumor is:

| Label | Class | Meaning |
|-------|-------|---------|
| `0` | Malignant | Cancerous |
| `1` | Benign | Non-cancerous |

The model is trained using the **Breast Cancer Wisconsin Diagnostic Dataset** available directly from `sklearn.datasets`.

---

## 📂 Dataset Information

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Source:** [`sklearn.datasets.load_breast_cancer()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

### Dataset Details

| Feature | Value |
|---------|-------|
| Total Samples | 569 |
| Total Features | 30 |
| Target Classes | Malignant / Benign |

### Features Include

Each of the following features provides **mean**, **standard error**, and **worst** values (30 features total):

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- ✅ Train-test split
- ✅ Feature standardization using `StandardScaler`
- ✅ Data normalization before training

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)
```

---

## 🧠 Neural Network Architecture

The model was implemented using the **TensorFlow/Keras Sequential API**.

```python
import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2,  activation='sigmoid')
])
```

### Layer Explanation

| Layer | Output Shape | Purpose |
|-------|-------------|---------|
| `Flatten` | (30,) | Converts input into 1D format |
| `Dense (20, ReLU)` | (20,) | Hidden layer for learning patterns |
| `Dense (2, Sigmoid)` | (2,) | Output layer for binary classification |

---

## ⚙️ Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

| Configuration | Value |
|--------------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |

---

## 🚀 Model Training

The model was trained for **10 epochs** with a **10% validation split**:

```python
history = model.fit(
    X_train_std,
    Y_train,
    validation_split=0.1,
    epochs=10
)
```

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | ~94% |
| Test Loss | Low |

> The model achieved strong accuracy on unseen test data, demonstrating good generalization performance.

---

## 🔍 Making Predictions

Example prediction on new input data:

```python
import numpy as np

# Standardize input
input_data_std = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_std)
prediction_label = [np.argmax(prediction)]

if prediction_label[0] == 0:
    print('The tumor is Malignant')
else:
    print('The tumor is Benign')
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Core programming language |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Numerical computations |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Data manipulation |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c) | Data visualization |
| ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Dataset & preprocessing |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | Neural network framework |
| ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) | High-level model API |
| ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white) | Interactive notebook environment |

---

## 📦 Installation

**1. Clone the repository:**

```bash
git clone https://github.com/Lucifer2987/breast-cancer-classification.git
cd breast-cancer-classification
```

**2. (Optional) Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## ▶️ Running the Project

**Open the notebook:**

```bash
jupyter notebook
```

Run the cells sequentially to:

1. 📥 Load the dataset
2. 🧹 Preprocess the data
3. 🧠 Build & compile the model
4. 🚀 Train the model
5. 📊 Evaluate performance
6. 🔍 Predict tumor class

---

## 📁 Project Structure

```
Breast-Cancer-Classification/
│
├── Breast_Cancer_Classification_with_Neural_Network.ipynb   # Main notebook
├── requirements.txt                                          # Python dependencies
└── README.md                                                 # Project documentation
```

---

## 📊 Future Improvements

Planned enhancements for future versions:

- [ ] Add `Dropout` layers to reduce overfitting
- [ ] Hyperparameter tuning (learning rate, epochs, batch size)
- [ ] Deeper model architecture exploration
- [ ] Confusion matrix & ROC curve visualization
- [ ] Cross-validation for robust evaluation
- [ ] Deploy using **Flask** or **Streamlit**
- [ ] Export model as `.h5` or `SavedModel` format

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
<p align="center">
  Made with ❤️ using TensorFlow & Keras
</p>
