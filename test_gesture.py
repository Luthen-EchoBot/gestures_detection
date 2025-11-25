import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_finger_states(landmarks):
    """
    Retourne une liste [index, middle, ring, pinky] où
    1 = doigt levé, 0 = doigt plié.
    """
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    states = []
    for tip_id, pip_id in zip(finger_tips, finger_pips):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        states.append(1 if tip.y < pip.y else 0)  # y plus petit = plus haut
    return states  # [index, middle, ring, pinky]


def pattern_confidence(pattern, states):
    """
    Calcule une confiance (0–100) selon le nombre de doigts
    qui correspondent au pattern.
    pattern / states : listes de 0/1 de taille 4.
    """
    mismatches = sum(p != s for p, s in zip(pattern, states))
    conf = 100 - mismatches * 25  # 4 doigts -> 25% par doigt
    return max(conf, 0)


def classify_gesture(landmarks):
    """
    Renvoie (label, confidence) parmi :
      Open_Palm, Closed_Fist, Victory, Pointing_Up, Other
    """
    states = get_finger_states(landmarks)  # [index, middle, ring, pinky]

    patterns = {
        "Open_Palm":   [1, 1, 1, 1],
        "Closed_Fist": [0, 0, 0, 0],
        "Victory":     [1, 1, 0, 0],
        "Pointing_Up": [1, 0, 0, 0],
    }

    # calcule la confiance pour chaque pattern
    best_label = None
    best_conf = -1
    for label, pattern in patterns.items():
        conf = pattern_confidence(pattern, states)
        if conf > best_conf:
            best_label = label
            best_conf = conf

    # seuil : si la meilleure confiance est faible, on classe en "Other"
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
        print("Camera error")
        return

    window_name = "Gesture Test"

    while True:
        ret, img = cap.read()
        if not ret:
            print("Frame grab failed")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        gesture = "None"
        confidence = 0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            gesture, confidence = classify_gesture(hand.landmark)

        cv2.putText(
            img,
            f"Gesture: {gesture} ({confidence}%)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, img)

        # quitter si fenêtre fermée ou touche 'q'
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
