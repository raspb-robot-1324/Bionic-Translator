#!/usr/bin/env python3

import sounddevice as sd
import numpy as np
import time

# Make sure this matches the device string you used in your main script!
DEVICE_ID = 0

def print_volume(indata, frames, time_info, status):
    # Calculate the exact volume level (RMS)
    volume = np.sqrt(np.mean(indata**2))
    
    # Create a visual bar so you can see spikes easily
    bar_length = int(volume * 500)
    visual_bar = "|" * bar_length
    
    # Print the exact number formatted to 4 decimal places
    print(f"Volume: {volume:.4f} {visual_bar}")

print("--- CALIBRATION DU MICROPHONE ---")
print("1. Restez silencieux pour mesurer le bruit ambiant de la salle.")
print("2. Parlez normalement pour voir votre volume de voix.")
print("Appuyez sur Ctrl+C pour quitter.\n")

try:
    with sd.InputStream(device=DEVICE_ID, channels=1, callback=print_volume):
        while True:
            # Keeps the script running without maxing out the Pi's CPU
            time.sleep(0.1) 
except KeyboardInterrupt:
    print("\nCalibration terminée.")
except Exception as e:
    print(f"Erreur: {e}")
