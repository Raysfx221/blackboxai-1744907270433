from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('model.h5')  # Load your trained CNN model here

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])  # Get the index of the highest prediction
    return jsonify({'class_index': class_index})

if __name__ == '__main__':
    app.run(debug=True)
