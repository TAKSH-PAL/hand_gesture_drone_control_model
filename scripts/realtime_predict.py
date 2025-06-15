import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque, Counter

# Load model and label encoder
model = tf.keras.models.load_model("models/gesture_model_mediapipe.h5")
with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Gesture smoothing setup
prediction_history = deque(maxlen=10)  # adjust for smoother or faster response
CONFIDENCE_THRESHOLD = 0.85

# Webcam feed
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 'q' to quit.")

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

            # Extract landmark features
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            # Predict gesture
            prediction = model.predict(np.array([data]))[0]
            confidence = np.max(prediction)

            if confidence > CONFIDENCE_THRESHOLD:
                gesture = encoder.inverse_transform([np.argmax(prediction)])[0]
                prediction_history.append(gesture)

    # Show most common gesture in recent frames
    if len(prediction_history) == prediction_history.maxlen:
        most_common = Counter(prediction_history).most_common(1)[0][0]
        cv2.putText(frame, f"Gesture: {most_common}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
