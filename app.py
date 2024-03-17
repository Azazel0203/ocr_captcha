from flask import Flask, request, jsonify
from PIL import Image
import os

from src.pipeline.prediction_pipeline import PredictionPipline

app = Flask(__name__)

@app.route('/')
def index():
    return open('templates\index2.html').read()

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'})

    # Save the uploaded image to a temporary file
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)

    # Perform OCR on the uploaded image
    extracted_text = perform_ocr(temp_image_path)

    # Delete the temporary image file
    os.remove(temp_image_path)

    return jsonify({'text': extracted_text, 'image_path': temp_image_path})

def perform_ocr(image_path):
    print("In ocr")
    obj = PredictionPipline('artifact\\model.weights.h5')
    print("loaded Model")
    text = obj.predict(image_path)
    print(text)
    return text

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)