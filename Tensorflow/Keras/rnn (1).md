# 🔁 Recurrent Neural Networks (RNNs)

RNNs are a class of neural networks for modeling sequential data, where the output from previous steps is fed as input to the current step.

## 🔍 Use Cases
- Language Modeling
- Text Generation
- Sequence Prediction

## 🧪 RNN in Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential([
    SimpleRNN(50, input_shape=(10, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

import numpy as np
x_train = np.random.rand(100, 10, 1)
y_train = np.random.rand(100, 1)

model.fit(x_train, y_train, epochs=5)
```
