import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# ----------------------------
# Initialize YOLO for person detection
# ----------------------------
model = YOLO("yolo11n.pt")
print("YOLO ready for detection")

# ----------------------------
# Initialize MediaPipe for hand gesture recognition
# ----------------------------
mp_hands = mp.solutions.hands

def get_finger_states(landmarks):
    """Return a list [index, middle, ring, pinky] where 1 = finger up, 0 = finger down"""
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    states = []
    for tip_id, pip_id in zip(finger_tips, finger_pips):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        states.append(1 if tip.y < pip.y else 0)
    return states

def pattern_confidence(pattern, states):
    """Compute confidence (0-100) of gesture matching pattern"""
    mismatches = sum(p != s for p, s in zip(pattern, states))
    conf = 100 - mismatches * 25  # 4 fingers, 25% penalty per mismatch
    return max(conf, 0)

def classify_gesture(landmarks):
    """Return gesture label and confidence based on finger states"""
    states = get_finger_states(landmarks)
    patterns = {
        "Open_Palm":   [1, 1, 1, 1],
        "Closed_Fist": [0, 0, 0, 0],
        "Victory":     [1, 1, 0, 0],
        "Pointing_Up": [1, 0, 0, 0],
    }
    best_label = None
    best_conf = -1
    for label, pattern in patterns.items():
        conf = pattern_confidence(pattern, states)
        if conf > best_conf:
            best_label = label
            best_conf = conf
    if best_conf < 50:
        return "Other", best_conf
    else:
        return best_label, best_conf

# ----------------------------
# Video capture setup
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

person_id_counter = 0
person_ids = {}  # dictionary: bounding box -> ID

print("Detecting people and gestures. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        height, width, _ = frame.shape
        frame_center_x = width / 2

        # ----------------------------
        # Detect persons with YOLO
        # ----------------------------
        results = model.track(frame, classes=[0], stream=False, persist=True)
        persons = []

        for result in results:
            boxes = result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.numpy()
            for i, box in enumerate(boxes):
                conf = confidences[i]
                x1, y1, x2, y2 = box.astype(int)
                bbox_center_x = (x1 + x2) / 2
                distance_center = abs(bbox_center_x - frame_center_x)

                # Assign ID to person
                bbox_key = (x1, y1, x2, y2)
                if bbox_key not in person_ids:
                    person_ids[bbox_key] = person_id_counter
                    person_id_counter += 1
                pid = person_ids[bbox_key]

                person_info = {
                    "id": pid,
                    "bbox": [x1, y1, x2, y2],
                    "distance_center": distance_center,
                    "confidence": float(conf)
                }

                # ----------------------------
                # Crop person and detect gestures
                # ----------------------------
                person_crop = frame[y1:y2, x1:x2]
                person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(person_crop_rgb)

                gestures = []
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        gesture_label, gesture_conf = classify_gesture(hand_landmarks.landmark)
                        gestures.append({
                            "person_id": pid,
                            "gesture": gesture_label,
                            "gesture_confidence": gesture_conf
                        })

                person_info["gestures"] = gestures
                persons.append(person_info)

        # ----------------------------
        # Output information
        # ----------------------------
        for p in persons:
            print(f"Person {p['id']} | BBox: {p['bbox']} | Distance to center: {p['distance_center']:.1f}px | Confidence: {p['confidence']:.2f}")
            for g in p["gestures"]:
                print(f"    Gesture: {g['gesture']} | Confidence: {g['gesture_confidence']}% | Person ID: {g['person_id']}")

        time.sleep(0.1)  # avoid flooding console

except KeyboardInterrupt:
    print("\nDetection stopped by user")

finally:
    cap.release()
    hands.close()
