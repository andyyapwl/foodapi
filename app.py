from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import onnxruntime as ort
from torchvision import transforms
import os
import requests

app = Flask(__name__)

# Download ONNX model from Hugging Face (once)
ONNX_MODEL_URL = "https://huggingface.co/andyywl/food-classifier-onnx/resolve/main/food_classifier.onnx"
ONNX_MODEL_PATH = "food_classifier.onnx"

if not os.path.exists(ONNX_MODEL_PATH):
    print("Downloading ONNX model...")
    r = requests.get(ONNX_MODEL_URL)
    with open(ONNX_MODEL_PATH, 'wb') as f:
        f.write(r.content)
    print("Model downloaded.")

# Load class labels
with open("class_labels.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Load ONNX model
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).numpy()  # shape: [1, 3, 224, 224]

    # Inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    predicted_idx = np.argmax(ort_outs[0])
    predicted_label = class_labels[predicted_idx]

    return jsonify({'predicted_category': predicted_label})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=False)
