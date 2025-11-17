import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Visualize predictions
sample_indices = np.random.choice(len(X_test), 5)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for idx, i in enumerate(sample_indices):
    image = X_test[i].reshape(28, 28)
    pred = model.predict(image.reshape(1, 28, 28, 1))
    pred_label = np.argmax(pred)
    axes[idx].imshow(image, cmap='gray')
    axes[idx].set_title(f"True: {y_test[i].argmax()}\nPred: {pred_label}")
    axes[idx].axis('off')
plt.show()