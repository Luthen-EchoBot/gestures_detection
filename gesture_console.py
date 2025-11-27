import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands


def get_finger_states(landmarks):
    """Retourne [index, middle, ring, pinky] avec 1 = doigt levé, 0 = plié."""
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    states = []
    for tip_id, pip_id in zip(finger_tips, finger_pips):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        states.append(1 if tip.y < pip.y else 0)
    return states


def pattern_confidence(pattern, states):
    """Confiance 0–100 en fonction du nombre de doigts qui matchent le pattern."""
    mismatches = sum(p != s for p, s in zip(pattern, states))
    conf = 100 - mismatches * 25  # 4 doigts -> 25% par doigt
    return max(conf, 0)


def classify_gesture(landmarks):
    """
    Renvoie (label, confidence) parmi :
    Open_Palm, Closed_Fist, Victory, Pointing_Up, Other
    """
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


def main():
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error (impossible d'ouvrir /dev/video0)")
        return

    last_print_time = 0.0
    last_label = None
    last_conf = 0

    print("Détection de gestes en cours. Ctrl+C pour arrêter.\n")

    try:
        while True:
            ret, img = cap.read()
            if not ret:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            label = "None"
            conf = 0

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                label, conf = classify_gesture(hand.landmark)

            now = time.time()

            # Affiche au maximum une fois par seconde, ou si le geste change
            if (now - last_print_time) > 1.0 or label != last_label:
                print(f"[{time.strftime('%H:%M:%S')}] Gesture: {label} ({conf}%)")
                last_print_time = now
                last_label = label
                last_conf = conf

    except KeyboardInterrupt:
        print("\nArrêt demandé (Ctrl+C).")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
