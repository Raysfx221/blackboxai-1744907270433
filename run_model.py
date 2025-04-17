import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('channa_model.keras')

# Define the categories
categories = ['channa_andrao', 'channa_barca', 'channa_bleheri']

# Function to predict the category of an image
def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to match model input
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize
    prediction = model.predict(image)
    predicted_class = categories[np.argmax(prediction)]
    return predicted_class

# Test the model with images in the dataset
for category in categories:
    path = os.path.join('dataset', category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        predicted_class = predict_image(img_path)
        print(f'Image: {img}, Predicted Class: {predicted_class}')
