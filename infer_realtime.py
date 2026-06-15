import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import argparse
from utils import extract_keypoints

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="models/sign_model_mlp_saved")
parser.add_argument("--threshold", type=float, default=0.0)
parser.add_argument("--use_tts", action="store_true")
args = parser.parse_args()

# --- Load Model ---
print("[INFO] Loading keypoint MLP model...")
try:
    model = tf.keras.models.load_model(args.model, compile=False)
    print(f"[SUCCESS] Model loaded. Expecting input shape: {model.input_shape}")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    exit()

# --- Load Classes ---
try:
    with open("models/classes.txt") as f:
        classes = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("[ERROR] 'models/classes.txt' not found.")
    exit()

# --- Text to Speech ---
tts = pyttsx3.init() if args.use_tts else None
last_spoken_time = 0  # FIXED debounce variable

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# --- Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

print("[INFO] Webcam started (press 'q' to quit)")

with mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            # --- Extract keypoints ---
            input_data = extract_keypoints(results, max_hands=2)

            # --- Predict ---
            preds = model.predict(input_data, verbose=0)[0]
            idx = np.argmax(preds)
            conf = np.max(preds)
            print(f"Prediction: {classes[idx]} (Confidence: {conf*100:.1f}%)")

            # --- Display result ---
            if conf > args.threshold:
                label = f"{classes[idx]} ({conf*100:.1f}%)"

                cv2.putText(
                    frame, label,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2
                )

                # --- TTS (fixed debounce) ---
                if tts:
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    if current_time - last_spoken_time > 2:
                        tts.say(classes[idx])
                        tts.runAndWait()
                        last_spoken_time = current_time

            # --- Draw landmarks ---
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Sign Language Detector (Keypoints)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Session ended.")