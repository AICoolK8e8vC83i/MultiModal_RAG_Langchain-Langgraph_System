# ⏳ Time Series Forecasting with LSTM (PyTorch)

LSTM can be used to predict future values in a time series.

## 🧪 Predicting Next Value in Sequence

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])  # Last output

model = TimeSeriesLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate toy sinusoidal data
x_data = np.linspace(0, 100, 1000)
y_data = np.sin(x_data)
seq_length = 50

def create_dataset(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_dataset(y_data, seq_length)
X = torch.FloatTensor(X).unsqueeze(-1)  # (samples, seq_len, 1)
y = torch.FloatTensor(y).unsqueeze(-1)

for epoch in range(10):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
