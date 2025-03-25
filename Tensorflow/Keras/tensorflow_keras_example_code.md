# TensorFlow + Keras Example Code: Image Classifier

## 🔍 Core Concept

TensorFlow with Keras provides a high-level API to build and train deep learning models. Below is a minimal example of a CNN classifier using the CIFAR-10 dataset.

## 🧠 Key Concepts

- Loading datasets with `tf.keras.datasets`
- Creating models with the Sequential API
- Compiling and fitting models with standard training loops

## 🚀 Sample Code

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile and train model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

## 📌 Prompt Examples

- “How do I build a CNN using TensorFlow/Keras?”
- “What’s the easiest way to classify CIFAR-10 images?”

## 🖼️ Related Image/Video Ideas

- Architecture visualization of the CNN layers.
- Training accuracy and loss graphs across epochs.

## 🗣️ Audio Prompt Hook

- “What is Keras used for in deep learning?”
- “How do I visualize this TensorFlow model?”
