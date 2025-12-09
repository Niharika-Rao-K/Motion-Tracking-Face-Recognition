#!/usr/bin/env python3
import cv2
import os
import numpy as np
import time

# ---------------- CONFIG ----------------
CASCADE_PATH = "/home/raspberry/haarcascade_frontalface_default.xml"
DATASET_PATH = "/home/raspberry/dataset"
MODEL_PATH = "/home/raspberry/face_model.yml"
CAM_DEVICE = 0
NUM_IMAGES = 30
IMG_SIZE = (200, 200)  # all faces resized to this
MIN_FACE_SIZE = 50
# ---------------------------------------

# Load face cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise IOError("Failed to load cascade!")

# Ask user for name
person_name = input("Enter the person name: ").strip()
if not person_name:
    raise ValueError("Person name cannot be empty!")

# Create folder for this person
person_dir = os.path.join(DATASET_PATH, person_name)
os.makedirs(person_dir, exist_ok=True)

# Start camera
cap = cv2.VideoCapture(CAM_DEVICE)
time.sleep(1.0)

faces = []
count = 0
print(f"[INFO] Starting capture for {person_name}. Press 'q' to quit.")

while count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )

    for (x, y, w, h) in detected:
        # Ignore very small faces
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        # Make sure rectangle is inside frame
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face_img = gray[y1:y2, x1:x2]

        # Resize to fixed size
        try:
            face_img = cv2.resize(face_img, IMG_SIZE)
            faces.append(face_img)

            # Save image
            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1

            cv2.imshow("Capture", face_img)
        except cv2.error:
            continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

faces = np.array(faces)
print(f"[INFO] Captured {len(faces)} images for {person_name}")

# --- Train LBPH recognizer on entire dataset ---
print("[INFO] Training LBPH recognizer on dataset...")

label_dict = {}
faces_all = []
labels_all = []
label_id = 0

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)
    if os.path.isdir(person_path):
        label_dict[label_id] = person
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            faces_all.append(img)
            labels_all.append(label_id)
        label_id += 1

faces_all = np.array(faces_all)
labels_all = np.array(labels_all)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces_all, labels_all)
recognizer.save(MODEL_PATH)

print(f"[INFO] Training complete. Model saved to {MODEL_PATH}")
print(f"[INFO] Labels mapping: {label_dict}")
