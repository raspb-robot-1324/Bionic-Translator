import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# --- PART 1: THE ROBOT BRAIN (Mapping) ---
class VirtualHandEngine:
    def __init__(self):
        # 0 = Closed (Fist), 100 = Open (Straight)
        self.current_finger_states = [0, 0, 0, 0, 0] # Thumb, Index, Mid, Ring, Pinky

        # THE "QUEBEC-ROBOTIQUE" MODIFIED ALPHABET
        self.alphabet_map = {
            'A': [0, 0, 0, 0, 0], 'B': [0, 100, 100, 100, 100],
            'C': [50, 50, 50, 50, 50], 'D': [0, 100, 0, 0, 0],
            'E': [0, 0, 0, 0, 0], 'F': [0, 0, 100, 100, 100],
            'G': [0, 100, 0, 0, 0], 'H': [0, 100, 100, 0, 0],
            'I': [0, 0, 0, 0, 100], 'J': [0, 0, 0, 0, 100], # Static J
            'K': [0, 100, 100, 0, 0], 'L': [100, 100, 0, 0, 0],
            'M': [0, 50, 50, 50, 0], # Draped
            'N': [0, 50, 50, 0, 0],  # Draped
            'O': [0, 0, 0, 0, 0],
            'P': [0, 100, 100, 0, 0], # Forward K
            'Q': [0, 100, 0, 0, 0],   # Forward G
            'R': [0, 100, 100, 0, 0],
            'S': [100, 0, 0, 0, 0],   # Thumb Up Fist
            'T': [100, 0, 0, 0, 0],   # Thumb Up Fist
            'U': [0, 100, 100, 0, 0], 'V': [0, 100, 100, 0, 0],
            'W': [0, 100, 100, 100, 0], 'X': [0, 50, 0, 0, 0],
            'Y': [100, 0, 0, 0, 100],
            'Z': [100, 100, 50, 0, 0], # Modified L
            ' ': [0, 0, 0, 0, 0]
        }

    def get_target_state(self, letter):
        """Returns the 5-finger array for a letter"""
        char = letter.upper()
        if char in self.alphabet_map:
            return self.alphabet_map[char]
        return [0, 0, 0, 0, 0] # Default to Fist

# --- PART 2: THE VISION SYSTEM ---
class SignDetector:
    def __init__(self, knowledge_file="asl_knowledge.csv"):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.knowledge_base = []
        self.load_knowledge(knowledge_file)

    def load_knowledge(self, filename):
        if not os.path.exists(filename): return
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                try:
                    self.knowledge_base.append((row[0], np.array([float(x) for x in row[1:]])))
                except: continue

    def predict(self, landmarks):
        if not self.knowledge_base: return "?"
        # Flatten landmarks
        base_x, base_y = landmarks[0].x, landmarks[0].y
        current_vec = []
        for lm in landmarks:
            current_vec.extend([lm.x - base_x, lm.y - base_y])
        current_vec = np.array(current_vec)

        best_label = "?"
        min_dist = 0.7 # Threshold
        for label, saved_vec in self.knowledge_base:
            dist = np.linalg.norm(current_vec - saved_vec)
            if dist < min_dist:
                min_dist = dist
                best_label = label
        return best_label

# --- PART 3: THE MAIN LOOP ---
def draw_dashboard(frame, finger_states, detected_char, last_typed_word):
    # Draw Background Panel for "Virtual Robot"
    cv2.rectangle(frame, (450, 0), (640, 200), (30, 30, 30), -1)
    cv2.putText(frame, "VIRTUAL ROBOT", (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Draw 5 Energy Bars (Thumb -> Pinky)
    labels = ["T", "I", "M", "R", "P"]
    for i, state in enumerate(finger_states):
        # Bar Background
        cv2.rectangle(frame, (470 + i*30, 150), (490 + i*30, 50), (50, 50, 50), -1)
        # Bar Fill (Height depends on state 0-100)
        fill_height = int(state)
        color = (0, 0, 255) if state < 20 else (0, 255, 0) # Red if closed, Green if open

        start_point = (470 + i*30, 150)
        end_point = (490 + i*30, 150 - fill_height)
        cv2.rectangle(frame, start_point, end_point, color, -1)
        cv2.putText(frame, labels[i], (475 + i*30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Draw Text Display
    cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
    cv2.putText(frame, f"DETECTED: {detected_char}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"WORD: {last_typed_word}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    robot = VirtualHandEngine()
    vision = SignDetector("asl_knowledge.csv")
    cap = cv2.VideoCapture(0)

    current_detected_letter = "?"
    typed_word = ""
    hold_start = 0
    last_stable_char = "?"

    print("--- VIRTUAL MODE ---")
    print("1. Show hand to Camera -> See recognized letter.")
    print("2. Press 'KEYBOARD' letters -> See Virtual Robot move.")
    print("3. Press 'q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- VISION PIPELINE ---
        results = vision.hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                vision.mp_draw.draw_landmarks(frame, hand_lms, vision.mp_hands.HAND_CONNECTIONS)
                current_detected_letter = vision.predict(hand_lms.landmark)

                # Simple Holding Logic
                if current_detected_letter == last_stable_char:
                    if time.time() - hold_start > 1.0 and current_detected_letter != "?":
                        if not typed_word.endswith(current_detected_letter):
                            typed_word += current_detected_letter
                            # Make the Virtual Robot mimic the detected sign!
                            robot.current_finger_states = robot.get_target_state(current_detected_letter)
                            hold_start = time.time() + 1.0 # Cooldown
                else:
                    last_stable_char = current_detected_letter
                    hold_start = time.time()

        # --- KEYBOARD INPUT (To Test Robot Logic) ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif 97 <= key <= 122: # a-z
            char = chr(key).upper()
            typed_word += char
            robot.current_finger_states = robot.get_target_state(char)
        elif key == 8: # Backspace
            typed_word = typed_word[:-1]

        # Draw Dashboard
        draw_dashboard(frame, robot.current_finger_states, current_detected_letter, typed_word)

        cv2.imshow("Virtual Robot Test", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()