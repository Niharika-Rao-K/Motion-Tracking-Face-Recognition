#!/usr/bin/env python3
import cv2
import os
import time
import numpy as np
import pigpio
# --- Build label_dict from dataset folders ---
import os

DATASET_PATH = "/home/raspberry/dataset"  # make sure this matches your folder
label_dict = {}
label_id = 0

for person in sorted(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person)
    if os.path.isdir(person_path):
        label_dict[label_id] = person
        label_id += 1

print("[INFO] Loaded label mapping:", label_dict)

# ---------------- CONFIG ----------------
VIDEO_DEVICE = 0
FRAME_W, FRAME_H = 640, 480

PAN_PIN, TILT_PIN = 17, 27
PAN_CENTER, TILT_CENTER = 90, 90
PAN_MIN, PAN_MAX = 40, 140
TILT_MIN, TILT_MAX = 40, 140

KP_FAST = 0.02    # faster servo response
MOVE_DEADZONE = 15
MOTION_THRESH = 40
MIN_MOTION_AREA = 6000
MIN_FACE_AREA = 1500
NO_MOTION_THRESHOLD = 30  # frames

CASCADE_PATH = "/home/raspberry/haarcascade_frontalface_default.xml"
MODEL_PATH = "/home/raspberry/face_model.yml"
LOGS_PATH = "/home/raspberry/logs"

FACE_IMG_SIZE = (200, 200)
RECOGNITION_CONFIDENCE = 70  # lower = stricter
# ---------------------------------------

def angle_to_pulse_width(angle):
    return int(500 + (angle / 180.0) * 2000)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# --- Initialize pigpio ---
pi = pigpio.pi()
if not pi.connected:
    print("[ERROR] Failed to connect to pigpio daemon! Run: sudo pigpiod")
    exit(1)
print("[INFO] pigpio connected successfully")

# --- Load face cascade and recognizer ---
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("[ERROR] Could not load Haar Cascade!")
    exit(1)

if not os.path.exists(MODEL_PATH):
    print("[ERROR] No trained model found! Run your capture/training script first.")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)
print("[INFO] Loaded trained LBPH model")

# --- Camera setup ---
cap = cv2.VideoCapture(VIDEO_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
time.sleep(1.0)
if not cap.isOpened():
    print("[ERROR] Failed to open camera.")
    pi.stop()
    exit(1)

# --- Initialize servos ---
pan_angle, tilt_angle = float(PAN_CENTER), float(TILT_CENTER)
pi.set_servo_pulsewidth(PAN_PIN, angle_to_pulse_width(pan_angle))
pi.set_servo_pulsewidth(TILT_PIN, angle_to_pulse_width(tilt_angle))

frame_center_x, frame_center_y = FRAME_W // 2, FRAME_H // 2
prev_gray = None
no_motion_counter = 0

ensure_dir(os.path.join(LOGS_PATH, "recognized"))
ensure_dir(os.path.join(LOGS_PATH, "unrecognized"))

print("[INFO] Motion + Face Recognition started. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        motion_detected = False
        face_detected = False

        # --- MOTION DETECTION ---
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray_blur)
            thresh = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < MIN_MOTION_AREA:
                    continue
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                # --- Servo motion tracking ---
                error_x = cx - frame_center_x
                error_y = cy - frame_center_y

                # --- Servo motion tracking ---
                if abs(error_x) > MOVE_DEADZONE or abs(error_y) > MOVE_DEADZONE:
                # Calculate new target angles
                  target_pan = pan_angle - error_x * 0.03
                  target_tilt = tilt_angle + error_y * 0.03

                # Smooth transition (low-pass filter)
                  pan_angle = 0.8 * pan_angle + 0.2 * target_pan
                  tilt_angle = 0.8 * tilt_angle + 0.2 * target_tilt

                # Limit angles to range
                  pan_angle = np.clip(pan_angle, PAN_MIN, PAN_MAX)
                  tilt_angle = np.clip(tilt_angle, TILT_MIN, TILT_MAX)

                # Update servo pulsewidths
                  pi.set_servo_pulsewidth(PAN_PIN, angle_to_pulse_width(pan_angle))
                  pi.set_servo_pulsewidth(TILT_PIN, angle_to_pulse_width(tilt_angle))

                  time.sleep(0.02)  # small delay for stability



        prev_gray = gray_blur.copy()

        # --- FACE DETECTION + RECOGNITION ---
        if motion_detected:
         faces = face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
         if len(faces) > 0:
          face_detected = True
          (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
          if (w * h) > MIN_FACE_AREA:
            cx, cy = x + w // 2, y + h // 2
            face_gray = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_gray, FACE_IMG_SIZE)

            # --- Predict face ---
            label_id, confidence = recognizer.predict(face_resized)

            if confidence > RECOGNITION_CONFIDENCE:
                name = "Unknown"
            else:
                name = label_dict.get(label_id, f"Person {label_id}")

            label_text = f"{name} ({int(confidence)})"
            folder = "unrecognized" if name == "Unknown" else "recognized"
            log_path = os.path.join(LOGS_PATH, folder)
            os.makedirs(log_path, exist_ok=True)  # ensure folder exists

            # --- Save 3 face images ---
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            for i in range(1, 4):  # capture 3 images
                img_path = os.path.join(log_path, f"{label_text}_{timestamp}_{i}.jpg")
                cv2.imwrite(img_path, face_resized)
                time.sleep(0.2)

            # --- Draw info on frame ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_text}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # --- No motion handling ---
        if motion_detected:
            no_motion_counter = 0
        else:
            no_motion_counter += 1

        if no_motion_counter > NO_MOTION_THRESHOLD:
            pan_angle += (PAN_CENTER - pan_angle) * 0.05
            tilt_angle += (TILT_CENTER - tilt_angle) * 0.05
            pi.set_servo_pulsewidth(PAN_PIN, angle_to_pulse_width(pan_angle))
            pi.set_servo_pulsewidth(TILT_PIN, angle_to_pulse_width(tilt_angle))

        # --- Display frame ---
        cv2.putText(frame,
                    f"Motion: {'Yes' if motion_detected else 'No'} | Face: {'Yes' if face_detected else 'No'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Motion + Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pi.set_servo_pulsewidth(PAN_PIN, 0)
    pi.set_servo_pulsewidth(TILT_PIN, 0)
    pi.stop()
    print("[INFO] Cleanup complete")
