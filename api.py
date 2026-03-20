from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import json
import shutil
import os
import gdown
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import img_to_array
from insightface.app import FaceAnalysis

# ── Google Drive file IDs ─────────────────────────────────
DRIVE_FILES = {
    "hybrid_full_trained.keras": "1cWcN3VcXzWGNSssi48KuV9EGSkeF0b2s",
    "arcface_embeddings_train.npy": "1GjG_0DRKbby5APbQeIe_8PpFos6bMvir",
    "arcface_labels_train.npy": "1Nrx2298a5yGbg9UREqX5PgnfmZUVwNWf",
    "arcface_class_names_train.json": "1ym-eUKfdBPaKdylFUVA7tb7vWUtlO1Cr",
}

# ── Download model files if not present ──────────────────
def download_models():
    for filename, file_id in DRIVE_FILES.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)
            print(f"Downloaded {filename} ✅")
        else:
            print(f"{filename} already exists ✅")

download_models()

# ── Load model and data ───────────────────────────────────
print("Loading model... please wait")
model = load_model("hybrid_full_trained.keras")

with open("arcface_class_names_train.json", "r") as f:
    class_names = json.load(f)

arc_embeddings = np.load("arcface_embeddings_train.npy")
arc_labels = np.load("arcface_labels_train.npy")

face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded and ready! ✅")

# ── Helper functions ──────────────────────────────────────
def preprocess_image(image, size=(224, 224)):
    face = cv2.resize(image, size)
    face = face.astype("float32") / 255.0
    face = img_to_array(face)
    return np.expand_dims(face, axis=0)

def recognize_faces(image):
    results = []
    faces = face_app.get(image)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cnn_input = preprocess_image(crop)
        arc_input = np.expand_dims(face.embedding, axis=0)
        pred = model.predict([arc_input, cnn_input], verbose=0)[0]

        top_idx = np.argmax(pred)
        softmax_score = float(pred[top_idx])
        softmax_name = class_names[top_idx]

        sims = cosine_similarity(arc_input, arc_embeddings)[0]
        sim_top_idx = np.argmax(sims)
        sim_top_score = float(sims[sim_top_idx])
        sim_top_label = arc_labels[sim_top_idx]
        sim_top_name = class_names[sim_top_label]

        SOFTMAX_THRESHOLD = 0.90
        SIMILARITY_THRESHOLD = 0.45

        if softmax_name == sim_top_name and sim_top_score >= SIMILARITY_THRESHOLD and softmax_score >= SOFTMAX_THRESHOLD:
            final_name = softmax_name
        elif sim_top_score >= 0.60:
            final_name = sim_top_name
        else:
            final_name = "Unknown"

        top5 = [
            {"name": class_names[i], "confidence": round(float(pred[i]) * 100, 1)}
            for i in pred.argsort()[-5:][::-1]
        ]

        results.append({
            "name": final_name.replace("_", " "),
            "softmax_score": round(softmax_score * 100, 1),
            "cosine_score": round(sim_top_score * 100, 1),
            "top5": top5,
            "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
        })

    return results

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="Hybrid Face Recognition API",
    description="ArcFace + CNN hybrid model — upload a photo, get back who is in it.",
    version="1.0"
)

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Hybrid Face Recognition API",
        "model": "ArcFace + CNN trained on VGGFace2",
        "usage": "POST an image to /identify"
    }

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        image = cv2.imread(temp_path)
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not read image. Please upload a valid JPG or PNG."}
            )

        results = recognize_faces(image)

        if not results:
            return JSONResponse(content={
                "faces_detected": 0,
                "message": "No faces found in the image.",
                "results": []
            })

        return JSONResponse(content={
            "faces_detected": len(results),
            "results": results
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)