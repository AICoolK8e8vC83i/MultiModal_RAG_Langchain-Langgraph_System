# 🔁 Long Short-Term Memory Networks (LSTMs)

LSTMs are a type of Recurrent Neural Network designed to handle long-term dependencies in sequential data.

## 🔍 Use Cases
- Time Series Forecasting
- Sentiment Analysis
- Speech Recognition

## 🧪 LSTM Example in Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, input_shape=(10, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

import numpy as np
x_train = np.random.rand(100, 10, 1)
y_train = np.random.rand(100, 1)

model.fit(x_train, y_train, epochs=5)
```
