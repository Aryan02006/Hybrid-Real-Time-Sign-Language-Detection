import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
from glob import glob
from tqdm import tqdm

# --- Configuration ---
ROOT_DATA_DIR = 'dataset'
DATA_PATH = 'keypoint_data.csv'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# --- Load or Create DataFrame ---
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded existing data. Total rows: {len(df)}")
else:
    columns = [f'feature_{i}' for i in range(126)] + ['class']
    df = pd.DataFrame(columns=columns)
    print("Created new dataset file.")

# --- Keypoint Extraction ---
def extract_keypoints(results, max_hands=2):
    keypoints = []
    detected_hands = results.multi_hand_landmarks or []

    for hand_landmarks in detected_hands:
        for landmark in hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])

    expected_features = 21 * 3 * max_hands

    if len(keypoints) < expected_features:
        keypoints.extend([0.0] * (expected_features - len(keypoints)))

    return np.array(keypoints[:expected_features], dtype=np.float32)

# --- Collect Image Paths ---
image_paths = (
    glob(os.path.join(ROOT_DATA_DIR, '**', '*.jpg'), recursive=True) +
    glob(os.path.join(ROOT_DATA_DIR, '**', '*.jpeg'), recursive=True) +
    glob(os.path.join(ROOT_DATA_DIR, '**', '*.png'), recursive=True)
)

print(f"\nFound {len(image_paths)} images.")

# --- Process Images ---
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

    new_rows = []

    for image_path in tqdm(image_paths, desc="Processing Images"):
        class_name = os.path.basename(os.path.dirname(image_path))

        frame = cv2.imread(image_path)
        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            keypoints = extract_keypoints(results)
            row_data = list(keypoints) + [class_name]
            new_rows.append(row_data)

# --- Save Data ---
if new_rows:
    df_new = pd.DataFrame(new_rows, columns=df.columns)
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"\n✅ Data processing complete. Total rows: {len(df)}")
else:
    print("\n⚠️ No hands detected in images.")