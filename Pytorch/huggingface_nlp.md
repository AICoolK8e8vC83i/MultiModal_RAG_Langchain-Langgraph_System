# 🧠 NLP Tokenization + HuggingFace Transformers

Load pre-trained transformers and tokenize text for NLP tasks.

## 🧪 Sentiment Classification with DistilBERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Tokenize input
text = "This AI system is absolutely brilliant!"
inputs = tokenizer(text, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)

print("Sentiment probs:", probs)
```
