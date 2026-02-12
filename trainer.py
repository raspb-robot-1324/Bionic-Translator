import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import socket
from sklearn.neighbors import KNeighborsClassifier
import os
import pyttsx3

engine = pyttsx3.init()

# --- Set Voice to French ---
voices = engine.getProperty('voices')
for voice in voices:
    # Look for "FR" or "French" in the voice metadata
    if "french" in voice.name.lower() or "fr_FR" in voice.id:
        engine.setProperty('voice', voice.id)
        break

engine.setProperty('rate', 160)    # French sounds better slightly slower
engine.setProperty('volume', 0.9)

# --- Network Configuration ---
PI_IP = "172.29.18.143"  # <--- CHANGE THIS TO YOUR PI'S ACTUAL IP ADDRESS
PI_PORT = 65432

# --- ML Configuration ---
DATA_FILE = "lsq_custom_data.csv"

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.7
)

def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    wrist = coords[0]
    coords = coords - wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0: coords = coords / max_dist
    return coords.flatten()

def load_model():
    if not os.path.exists(DATA_FILE): return None
    df = pd.read_csv(DATA_FILE)
    if len(df) < 5: return None

    X = df.drop('label', axis=1).values
    y = df['label'].values
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

def send_to_pi(word):
    """Sends the assembled word to the Raspberry Pi over Wi-Fi."""
    try:
        # Create a temporary socket to send the data
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0) # 2-second timeout so Mac doesn't freeze if Pi is offline
            s.connect((PI_IP, PI_PORT))
            s.sendall(word.encode('utf-8'))
            return True
    except Exception as e:
        print(f"\n❌ Network Error: Could not reach Pi at {PI_IP}. Details: {e}")
        return False

def main():
    model = load_model()
    if model is None:
        print("❌ Error: No training data found! Run the training script first.")
        return

    cap = cv2.VideoCapture(0)
    current_word = ""
    last_prediction = ""

    print("\n--- 💻 MAC VISION CONTROLLER READY ---")
    print("[SPACE]     -> Add current letter to word")
    print("[BACKSPACE] -> Delete last letter")
    print("[ENTER]     -> Send word to Pi over Wi-Fi")
    print("[ESC]       -> Quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = normalize_landmarks(hand_landmarks)

                last_prediction = model.predict([features])[0]
                cv2.putText(frame, f"Sign: {last_prediction.upper()}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(frame, f"Word: {current_word}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 200, 0), 3)
        cv2.imshow("Mac LSQ Vision", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27: # ESC
            break
        elif key == 32: # SPACEBAR
            if last_prediction:
                current_word += last_prediction.upper()
                print(f"Added '{last_prediction.upper()}'. Current word: {current_word}")
        elif key in (8, 127): # BACKSPACE
            current_word = current_word[:-1]
        elif key in (13, 10): # ENTER
            if current_word:
                print(f"\n🚀 Sending '{current_word}' to Raspberry Pi...")
                success = send_to_pi(current_word)
                engine.say(current_word)
                engine.runAndWait()
                if success:
                    print("✅ Sent successfully!")
                    current_word = "" # Clear the word only if successfully sent

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()