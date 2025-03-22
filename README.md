# Car Body Type Classification using CNN

This project applies a Convolutional Neural Network (CNN) architecture for identifying different car body types from images. The model is trained on labeled image data and achieves high classification accuracy.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

This car body type classification project uses deep learning techniques with CNNs. CNNs are ideal for image recognition tasks due to their ability to extract hierarchical features from image data. The model is trained to classify images into categories such as Convertible, Coupe, Hatchback, Pick-Up, SUV, Sedan, and VAN.

## Features

- Preprocessing and Augmentation of Image Data
- Custom CNN Model using TensorFlow and Keras
- Classification of Car Body Types
- Model Evaluation and Visualization

## Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV
- ImageDataGenerator for Augmentation

## Dataset

- The dataset consists of labeled images of different car body types.
- Categories include Convertible, Coupe, Hatchback, Pick-Up, SUV, Sedan, and VAN.
- Images are divided into training and validation sets.

## Model Training

A CNN model was built using TensorFlow and Keras with the following configuration:

- **Convolutional Layers**: 3 convolutional layers with ReLU activation
- **Pooling**: MaxPooling applied after each convolution layer
- **Fully Connected Layers**: Dense layer with 128 neurons
- **Output Layer**: Softmax activation with 7 neurons for classification

```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## Evaluation

The model was evaluated using accuracy and loss metrics. Visualizations of the training and validation accuracy were plotted to analyze model performance.

```python
loss, accuracy = model.evaluate(validation_data)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
```

## Results

- Model Accuracy: **92.4%**
- The model correctly classifies car body types across multiple categories.
- Visualization of accuracy and loss curves shows successful training with minimal overfitting.

This project demonstrates how CNNs can be used effectively for image classification tasks in the automotive industry and beyond.
