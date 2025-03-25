## Question: How do you code a deep learning neural network?

# 🧠 Deep Learning with PyTorch

## 🚀 Basic Feedforward Neural Network

```python
import torch
import torch.nn as nn
import torch.optim as optim

class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = FeedforwardNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training
x = torch.randn(1000, 100)
y = torch.randint(0, 10, (1000,))

for epoch in range(10):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
