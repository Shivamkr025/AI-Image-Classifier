from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
from PIL import Image
import torch

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

MODEL_NAME = "google/vit-base-patch16-224"
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("image-classification", model=MODEL_NAME, device=device)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    outputs = classifier(image, top_k=5)
    return jsonify({"predictions": outputs})

if __name__ == "__main__":
    app.run(debug=True)
