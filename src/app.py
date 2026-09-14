import os
from pathlib import Path
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from inference import load_model, predict
from config import APP_IMAGE_TYPES, APP_MAX_IMAGE_SIZE, CURRENT_MODEL

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR.parent / "checkpoints" / CURRENT_MODEL

ALLOWED_EXTENSIONS = APP_IMAGE_TYPES
MAX_CONTENT_LENGTH = APP_MAX_IMAGE_SIZE

if os.environ.get("RENDER"):
    torch.set_num_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(CHECKPOINT_PATH, device)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify(error="No file received."), 400
    if not allowed_file(file.filename):
        allowed_types = ", ".join(sorted(APP_IMAGE_TYPES)).upper()
        return jsonify(error=f"Unsupported file type. Use {allowed_types}."), 400
    try:
        pil_image = Image.open(file.stream)
        pil_image.load()
    except UnidentifiedImageError:
        return jsonify(error="File could not be read as an image."), 400

    results = predict(model, device, pil_image)
    ranked = sorted(results.items(), key=lambda item: item[1], reverse=True)
    top_label, top_confidence = ranked[0]

    return jsonify(
        results=ranked,
        top_label=top_label,
        top_confidence=top_confidence,
    )

if __name__ == "__main__":
    app.run(debug=True)