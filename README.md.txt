🖐️ Hand Gesture Recognition for Drone Control
This project uses MediaPipe, OpenCV, and TensorFlow to recognize hand gestures in real-time from a webcam feed. Recognized gestures can be mapped to control commands for a drone (e.g., DJI Tello).

📂 Folder Structure

hand_gesture_control/
├── data/
│   └── gestures.csv              # Collected landmark data
├── models/
│   ├── gesture_model_mediapipe.h5  # Trained ML model
│   └── label_encoder.pkl           # Encoder for gesture labels
├── scripts/
│   ├── collect_landmarks.py     # Collect labeled hand landmark data
│   ├── train_model.py           # Train gesture recognition model
│   └── realtime_predict.py      # Real-time gesture prediction from webcam
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

🚀 How It Works
MediaPipe extracts 21 hand landmarks per frame.

These 3D coordinates are used to train a classifier.

The model is trained on gesture-labeled landmark data.

In real-time, the model predicts the gesture and displays it on screen.

(Optional) Detected gestures can trigger drone commands.

🛠️ Setup Instructions
1. Clone the Repository

git clone https://github.com/your-username/hand_gesture_control.git
cd hand_gesture_control
2. Create a Virtual Environment

python -m venv venv
venv\Scripts\activate   # On Windows
3. Install Dependencies

pip install -r requirements.txt
📸 Step 1: Collect Gesture Data
Run this to collect labeled landmark data for one gesture:

python scripts/collect_landmarks.py
Adjust the GESTURE_LABEL variable inside the script before running.

Press s to save a sample, and q to quit.

Repeat for each gesture (e.g., fist, open_palm, left, right, etc.).

🧠 Step 2: Train the Model
After collecting all gesture samples:

python scripts/train_model.py
This will generate:

models/gesture_model_mediapipe.h5

models/label_encoder.pkl

🕵️ Step 3: Real-Time Gesture Detection
Use your webcam to test the model in real-time:

python scripts/realtime_predict.py
Predicted gestures will appear on the video frame.

Flickering is reduced using a smoothing buffer.

🛸 (Optional) Drone Control
If you're using a DJI Tello drone, you can extend realtime_predict.py to send commands using djitellopy:

pip install djitellopy
Example usage inside the frame loop:

if most_common == "left":
    tello.move_left(30)
elif most_common == "fist":
    tello.land()
📦 Dependencies
Listed in requirements.txt