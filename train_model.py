import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to the dataset
data_dir = 'dataset'
categories = ['channa_andrao', 'channa_barca', 'channa_bleheri']

# Load and preprocess images
def load_data():
    images = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)  # Assign a number to each category
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (128, 128))  # Resize images to 128x128
            images.append(image)
            labels.append(class_num)
    return np.array(images), np.array(labels)

# Load data
X, y = load_data()
X = X / 255.0  # Normalize the images

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10)

# Save the model
model.save('channa_model.keras')
