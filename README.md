# NTI-CNN-Flowers-Dataset 🌸

This project trains a Convolutional Neural Network (CNN) to classify flower images from the [Flowers Recognition dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).  
It uses **TensorFlow/Keras** for model training and **Streamlit** for deployment as a web application.

---

## 📂 Dataset
- Dataset: **Flowers Recognition** (from Kaggle).
- Contains images of **5 flower classes**:
  - Daisy
  - Dandelion
  - Rose
  - Sunflower
  - Tulip
- Images are resized to **320 × 240** for training.

---

## 🛠️ Data Preprocessing
- Used `ImageDataGenerator` for **data augmentation**:
  - Rescaling pixel values to `[0, 1]`
  - Random rotation, width/height shift, shear, zoom
  - Horizontal flipping
- Split dataset into:
  - **80% training**
  - **20% validation**

---

## 🏗️ Model Architecture
The CNN model was built using **Keras Sequential API**:

1. **Conv2D (64 filters, 3×3, ReLU, same padding)** → **MaxPooling2D**
2. **Conv2D (32 filters, 3×3, ReLU, same padding)** → **MaxPooling2D**
3. **Conv2D (1 filter, 3×3, ReLU, same padding)** → **MaxPooling2D**
4. **Flatten**
5. **Dense (32 units, ReLU)**
6. **Dense (16 units, ReLU)**
7. **Dense (32 units, ReLU)**
8. **Dense (5 units, Softmax)** → final classification layer (5 flower classes)

---

## ⚙️ Training
- **Optimizer**: Adam  
- **Loss function**: Categorical Crossentropy  
- **Metrics**: Accuracy  
- **Epochs**: 10  

---

## 💾 Saving the Model
After training, the model was saved in **Keras format**:

```python
model.save("CNN.keras")
