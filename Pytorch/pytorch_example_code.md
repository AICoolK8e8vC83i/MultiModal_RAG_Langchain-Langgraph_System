# PyTorch Example Code: Image Classifier

## 🔍 Core Concept

PyTorch is a deep learning framework known for its flexibility and dynamic computation graph. Here's a minimal example of a CNN for image classification using PyTorch.

## 🧠 Key Concepts

- Dataset loading with `torchvision.datasets` and `DataLoader`
- Defining a CNN using `nn.Module`
- Backpropagation and optimization with `torch.optim`

## 🚀 Sample Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Define simple CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 6 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training loop
for epoch in range(2): 
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## 📌 Prompt Examples

- “How do I define a CNN in PyTorch?”
- “Give me an example of training on CIFAR-10 using PyTorch.”

## 🖼️ Related Image/Video Ideas

- Diagram of the CNN architecture flow.
- Output image predictions and class probabilities.

## 🗣️ Audio Prompt Hook

- “What does backpropagation do in this PyTorch model?”
- “How do I switch this model to GPU?”
