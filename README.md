# Day & Night Image Classifier using Logistic Regression

## Overview

This project implements a **Day vs Night Image Classification System** using **Logistic Regression from scratch in NumPy**. The goal is to classify images as either **Day** or **Night** based on their visual features.

Unlike many machine learning projects that rely on high-level frameworks, this implementation focuses on understanding the fundamentals of machine learning by manually implementing the complete Logistic Regression pipeline, including data preprocessing, forward propagation, cost calculation, backward propagation, gradient descent optimization, and prediction.

The project demonstrates how a traditional machine learning algorithm can be applied to image classification tasks by converting images into numerical feature vectors and learning patterns directly from pixel values.

---

## Features

* Image classification using Logistic Regression
* Built entirely from scratch using NumPy
* Automatic image loading and preprocessing
* RGB image handling with OpenCV
* Dataset storage using HDF5 format
* Binary Cross-Entropy cost function
* Sigmoid activation function
* Gradient Descent optimization
* Training and testing accuracy evaluation
* End-to-end machine learning pipeline

---

## Project Workflow

### 1. Dataset Loading

Images are loaded from the dataset directory containing two classes:

```text
Datasets/
│
├── train/
│   ├── Day/
│   └── Night/
│
└── test/
    ├── Day/
    └── Night/
```

Each image is assigned a numerical label:

| Class | Label |
| ----- | ----- |
| Day   | 0     |
| Night | 1     |

---

### 2. Image Preprocessing

Each image undergoes the following preprocessing steps:

* Read using OpenCV
* Resize to 64 × 64 pixels
* Convert from BGR to RGB format
* Normalize pixel values to the range [0, 1]
* Flatten into a one-dimensional feature vector

After preprocessing, every image contains:

```text
64 × 64 × 3 = 12,288 features
```

---

### 3. Dataset Storage

The processed dataset is stored in an HDF5 file:

```text
data.h5
```

The file contains:

```text
train_set_x
train_set_y
test_set_x
test_set_y
```

This allows efficient storage and retrieval of training and testing data.

---

### 4. Logistic Regression Model

The model is implemented completely from scratch.

#### Forward Propagation

Computes the linear combination:

Z = wᵀX + b

The result is passed through the Sigmoid activation function to obtain probabilities.

#### Sigmoid Activation Function

The sigmoid function converts any real-valued number into a probability between 0 and 1.

It enables the model to perform binary classification by estimating the probability that an image belongs to the Night class.

---

### 5. Cost Function

The model uses **Binary Cross-Entropy Loss** to measure prediction error.

The objective of training is to minimize this loss function by adjusting the model parameters.

A lower cost indicates better predictions.

---

### 6. Optimization

The model uses **Gradient Descent** to learn optimal values of:

* Weights (w)
* Bias (b)

During each iteration:

1. Compute predictions
2. Calculate loss
3. Compute gradients
4. Update parameters
5. Repeat until convergence

---

### 7. Prediction

After training, the model predicts:

```text
Probability > 0.5  → Night
Probability ≤ 0.5 → Day
```

The final performance is measured using:

* Training Accuracy
* Testing Accuracy

---

## Technologies Used

* Python
* NumPy
* OpenCV
* H5Py
* Scikit-learn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/day-night-image-classifier.git
cd day-night-image-classifier
```

Install dependencies:

```bash
pip install numpy opencv-python h5py scikit-learn
```

---

## Running the Project

Execute the main Python file:

```bash
python main.py
```

The script will:

1. Load and preprocess images
2. Create the HDF5 dataset
3. Train the Logistic Regression model
4. Evaluate model performance
5. Display training and testing accuracy

---

## Example Output

```text
Loaded X shape: (1000, 64, 64, 3)
Loaded Y shape: (1000,)

Cost after 0: 0.6931
Cost after 100: 0.5214
Cost after 200: 0.4372
...

Final Results:
Train accuracy: 94.8%
Test accuracy: 91.5%
```

---

## Learning Outcomes

This project helped demonstrate:

* Fundamentals of Logistic Regression
* Image preprocessing techniques
* Binary classification concepts
* Sigmoid activation function
* Binary Cross-Entropy loss
* Forward and backward propagation
* Gradient Descent optimization
* Data handling using HDF5 files
* Building machine learning models without frameworks

---

## Future Improvements

Possible enhancements include:

* Support for larger datasets
* Hyperparameter tuning
* Feature engineering
* Multiclass classification
* Neural Network implementation
* Convolutional Neural Network (CNN) version
* Model deployment using Flask or Streamlit
* Real-time image prediction interface

---

## Author

**Anamta Gohar**

Computer Science Student | AI & Machine Learning Enthusiast

Built as part of a machine learning project to explore image classification and understand Logistic Regression from first principles.
