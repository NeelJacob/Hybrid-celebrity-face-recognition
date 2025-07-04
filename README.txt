# Hybrid Face Recognition System

This project implements a hybrid real-time face recognition system using **ArcFace embeddings** and a **custom CNN classifier**. It includes both a **Tkinter-based desktop GUI** for real-time webcam-based and image-upload-based recognition. The system supports features like Top-5 predictions, dynamic face registration, voice feedback, and Wikipedia summary integration.

---

## Features

- Real-time face recognition from webcam (GUI & Web)
- Image upload-based recognition (Web)
- ArcFace embedding + CNN hybrid model
- Top-1 and Top-5 prediction with confidence scores
- Dynamic face registration (via webcam or image upload)
- Wikipedia summary and image display
- Voice feedback using text-to-speech
- Automatic logging of recognized identities

---

## Project Structure

├── gui_launcher.py      # Tkinter GUI launcher (desktop app)
├── gui_Main.py           
├── hybrid_model/        # ArcFace + CNN model files
│   ├── hybrid_full_trained.keras
│   ├── arcface_embeddings_train.npy
│   ├── arcface_labels_train.npy
│   └── arcface_class_names_train.json
├── face_registry/       # Custom registered faces
│   ├── images/
│   └── embeddings/
├── logs/
│   └── bio_log.txt      # Log of recognized faces

## Install Required Libraries

pip install tensorflow keras opencv-python pyttsx3 wikipedia insightface numpy Pillow scikit-learn

## Launch the GUI
python gui_launcher.py

