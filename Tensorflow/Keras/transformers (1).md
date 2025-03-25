# 🔄 Transformers (NLP & Vision)

Transformers use attention mechanisms to process sequences in parallel, enabling state-of-the-art performance in NLP and CV tasks.

## 🔧 Key Components
- Self-Attention
- Positional Encoding
- Encoder & Decoder Layers

## 🧠 HuggingFace Transformers Example

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

inputs = tokenizer("Deep learning is powerful", return_tensors="tf")
outputs = model(**inputs)
print(outputs.logits)
```

## 📦 Vision Transformers

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification_bee.jpg'
image = Image.open(requests.get(url, stream=True).raw)

extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.argmax(-1))
```
