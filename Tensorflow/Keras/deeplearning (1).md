# 🧠 Deep Learning Overview

Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers (hence "deep") to model complex patterns in data.

## 🔧 Key Concepts
- **Neurons**: Inspired by the brain, process inputs with weights and biases.
- **Activation Functions**: ReLU, Sigmoid, Tanh.
- **Loss Functions**: MSE, Cross-Entropy.
- **Optimizers**: SGD, Adam.
- **Backpropagation**: Algorithm for training using gradient descent.

## 📦 Basic Neural Network (Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Dummy training
import numpy as np
x_train = np.random.random((1000, 100))
y_train = np.random.randint(10, size=(1000,))
model.fit(x_train, y_train, epochs=10, batch_size=32)
```
