import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# ----------------- Configuration -----------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = "plant_model.h5"   # put your Keras/TensorFlow model here
LABELS_PATH = "labels.txt"      # one label per line; index order must match model classes

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "change-this-to-a-secure-key"

# ----------------- Utilities -----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    # Model might expect shape (1, H, W, 3)
    return np.expand_dims(arr, axis=0)

# ----------------- Load model & labels -----------------
# Load model once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place your model there (Keras .h5).")

model = tf.keras.models.load_model(MODEL_PATH)
# Determine expected input size from model if possible:
try:
    input_shape = model.input_shape  # e.g., (None, 224, 224, 3)
    _, H, W, _ = input_shape
    TARGET_SIZE = (W, H)
except Exception:
    TARGET_SIZE = (224, 224)  # fallback

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}. Create one label per line.")

labels = load_labels(LABELS_PATH)

# ----------------- Routes -----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            flash("No selected file", "danger")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            # Preprocess and predict
            try:
                x = preprocess_image(save_path, TARGET_SIZE)
                preds = model.predict(x)[0]  # shape (num_classes,)
                # Get top-3
                top_indices = preds.argsort()[-3:][::-1]
                results = []
                for idx in top_indices:
                    label = labels[idx] if idx < len(labels) else f"Class {idx}"
                    confidence = float(preds[idx])
                    results.append({"label": label, "confidence": confidence})
                return render_template("index.html", filename=filename, results=results)
            except Exception as e:
                flash(f"Prediction error: {e}", "danger")
                return redirect(request.url)
        else:
            flash("Unsupported file format. Use png/jpg/jpeg.", "danger")
            return redirect(request.url)
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"), code=301)

# ----------------- Run -----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
