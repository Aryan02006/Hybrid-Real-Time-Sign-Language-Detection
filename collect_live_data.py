import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import argparse
from utils import extract_keypoints

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classname", type=str, required=True, help="The letter or number to record (e.g., A, B, 1)")
    parser.add_argument("-s", "--samples", type=int, default=100, help="Number of samples to collect")
    args = parser.parse_args()

    class_name = args.classname.upper()
    total_samples = args.samples
    DATA_PATH = 'keypoint_data.csv'

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Load existing data or create new
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Filter out synthetic data if any exists to keep dataset clean
        if 'feature_0' in df.columns:
            print(f"Loaded existing dataset with {len(df)} samples.")
    else:
        columns = [f'feature_{i}' for i in range(126)] + ['class']
        df = pd.DataFrame(columns=columns)
        print("Created a new dataset file.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print(f"\n--- Live Keypoint Collector ---")
    print(f"Target Class: {class_name}")
    print(f"Target Samples: {total_samples}")
    print(f"Instructions: Position your hand sign in the window, then press 's' to start recording.")

    recorded_rows = []
    recording = False

    with mp_hands.Hands(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        max_num_hands=2
    ) as hands:

        while len(recorded_rows) < total_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Draw status on the frame
            status_text = "READY - Press 's' to Start" if not recording else f"RECORDING: {len(recorded_rows)}/{total_samples}"
            color = (0, 255, 0) if not recording else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Class: {class_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            if results.multi_hand_landmarks:
                # Draw landmarks so the user knows hand is detected
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Record keypoints if recording started
                if recording:
                    keypoints = extract_keypoints(results, max_hands=2)[0]
                    row_data = list(keypoints) + [class_name]
                    recorded_rows.append(row_data)

            cv2.imshow("Live Gesture Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and not recording:
                recording = True
                print("Recording started...")
            elif key == ord('q'):
                print("Recording canceled.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(recorded_rows) > 0:
        new_df = pd.DataFrame(recorded_rows, columns=df.columns)
        # Clear synthetic data first to keep actual dataset fully accurate
        if os.path.exists(DATA_PATH) and len(df) > 0 and 'feature_0' in df.columns:
            # Check if dataset contains only synthetic data (e.g. exactly 3600 samples)
            if len(df) == 3600 and df['class'].nunique() == 36:
                print("Clearing synthetic data to start a clean, real-world hand dataset...")
                df = pd.DataFrame(columns=df.columns)

        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"\n[SUCCESS] Saved {len(recorded_rows)} samples for class '{class_name}'. Total dataset size: {len(df)}")
    else:
        print("\nNo samples recorded.")

if __name__ == '__main__':
    main()
