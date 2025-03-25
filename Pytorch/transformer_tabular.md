# 🔄 TransformerEncoder for Tabular & Multimodal Data

Transformers can be adapted for structured/tabular or multimodal (e.g., tabular + text/image) data.

## 🧪 Example: TransformerEncoder for Tabular Data

```python
import torch
import torch.nn as nn

class TabularTransformer(nn.Module):
    def __init__(self, num_features, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Linear(num_features, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embedding(x)  # (batch, seq_len, model_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch, model_dim)
        x = self.encoder(x)
        x = x.mean(dim=0)  # mean pooling over seq_len
        return self.classifier(x)

# Dummy data
x = torch.randn(32, 10, 20)  # (batch, seq_len, features)
model = TabularTransformer(num_features=20, model_dim=64, num_heads=4, num_layers=2, output_dim=2)
out = model(x)
print("Output shape:", out.shape)
```
