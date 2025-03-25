# 👁️ Attention Visualization (Self-Attention)

Visualizing self-attention scores helps understand where the model is focusing.

## 🧪 Attention Heatmap Example (Toy)

```python
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

# Dummy token sequence
seq_len = 5
model_dim = 8

q = torch.rand(seq_len, model_dim)
k = torch.rand(seq_len, model_dim)

# Dot product attention
scores = torch.matmul(q, k.transpose(0, 1)) / model_dim**0.5  # (seq_len, seq_len)
attn_weights = nn.Softmax(dim=-1)(scores)

# Heatmap
sns.heatmap(attn_weights.detach().numpy(), annot=True, cmap='viridis')
plt.title("Self-Attention Scores")
plt.xlabel("Key Positions")
plt.ylabel("Query Positions")
plt.show()
```
