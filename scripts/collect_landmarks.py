import cv2
import mediapipe as mp
import csv
import os

GESTURE_LABEL = "left"  # change this per gesture
SAVE_PATH = "data/gestures.csv"

os.makedirs("data", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Press 's' to save frame, 'q' to quit.")

with open(SAVE_PATH, mode='a', newline='') as f:
    writer = csv.writer(f)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    data = []
                    for lm in hand_landmarks.landmark:
                        data.extend([lm.x, lm.y, lm.z])
                    data.append(GESTURE_LABEL)
                    writer.writerow(data)
                    print(f"[INFO] Saved frame for '{GESTURE_LABEL}'.")

        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
