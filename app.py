from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow import keras
import io
import os

app = Flask(__name__)
CORS(app)

model = keras.models.load_model("model/en_model.keras")

def preprocess(img):
    img = img.convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = 1.0 - arr
    arr = arr.reshape(1, 28, 28, 1)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file.stream)
    x = preprocess(img)
    pred = model.predict(x)
    result = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return jsonify({"prediction": result, "confidence": round(confidence*100,2)})

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
