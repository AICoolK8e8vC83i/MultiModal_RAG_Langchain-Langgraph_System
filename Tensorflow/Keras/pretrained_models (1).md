# 🧠 Pretrained Models (Transfer Learning)

Pretrained models are deep learning models trained on large benchmark datasets (like ImageNet) and reused for other tasks.

## 🔧 Common Models
- ResNet50
- VGG16
- InceptionV3
- MobileNet
- EfficientNet

## 🧪 Example: Load ResNet50 in Keras

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Input

base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
