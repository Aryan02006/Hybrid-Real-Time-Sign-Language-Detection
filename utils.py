import cv2
import numpy as np

# Original functions (kept for completeness)
def expand_bbox(xmin, ymin, xmax, ymax, expansion=0.3, img_shape=(480, 640)):
    h, w = img_shape
    dx = int((xmax - xmin) * expansion)
    dy = int((ymax - ymin) * expansion)
    xmin = max(0, xmin - dx)
    ymin = max(0, ymin - dy)
    xmax = min(w, xmax + dx)
    ymax = min(h, ymax + dy)
    return xmin, ymin, xmax, ymax

def crop_and_resize(frame, bbox, size=(128, 128)):
    xmin, ymin, xmax, ymax = bbox
    crop = frame[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return cv2.resize(frame, size) 
    return cv2.resize(crop, size)

# NEW: Keypoint extraction function
def extract_keypoints(results, max_hands=2):
    """
    Extracts and normalizes 3D keypoints for up to max_hands (2).
    Pads with zeros if fewer than max_hands are detected (e.g., for single-hand signs).
    Returns a flattened array of shape (1, 126).
    """
    keypoints = []
    detected_hands = results.multi_hand_landmarks or []
    
    for hand_landmarks in detected_hands:
        for landmark in hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])

    # 21 landmarks * 3 coordinates/landmark * 2 hands = 126 features
    expected_features = 21 * 3 * max_hands 
    
    # Pad features with zeros if only one hand is detected
    if len(keypoints) < expected_features:
        keypoints.extend([0.0] * (expected_features - len(keypoints)))
    
    keypoints = keypoints[:expected_features]
    
    return np.array([keypoints], dtype=np.float32)