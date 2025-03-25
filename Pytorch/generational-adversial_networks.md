## Question: How do you code a Generational Adversial Network in Pytorch?

# 🧬 Generative Adversarial Networks (GANs) with PyTorch

GANs consist of two neural networks: the Generator and the Discriminator, competing in a zero-sum game.

## 🧪 GAN Example (Toy)

```python
import torch
import torch.nn as nn
import torch.optim as optim

latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(10):
    z = torch.randn(64, latent_dim)
    fake_images = G(z)
    real_labels = torch.ones(64, 1)
    fake_labels = torch.zeros(64, 1)

    # Train Discriminator
    real_images = torch.randn(64, 784)
    outputs_real = D(real_images)
    outputs_fake = D(fake_images.detach())
    d_loss = criterion(outputs_real, real_labels) + criterion(outputs_fake, fake_labels)
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    outputs = D(fake_images)
    g_loss = criterion(outputs, real_labels)
    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()

    print(f"Epoch {epoch+1} D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")
```
