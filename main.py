#!/usr/bin/env python3

import argparse
import queue
import sys
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from unidecode import unidecode
from whisper_cpp_python import Whisper
import re 
import subprocess
import alphabet

q = queue.Queue()

MODEL_PATH = "/home/othel/whisper_models/ggml-base.bin" 

DEVICE_ID = 0

# --- SILENCE DETECTION SETTINGS ---
SILENCE_THRESHOLD = 0.02  #TODO: TEST
SILENCE_DURATION = 1   
# ----------------------------------

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

parser = argparse.ArgumentParser(description="Whisper.cpp Local Transcription")
parser.add_argument("-r", "--samplerate", type=int, default=16000, help="sampling rate")
args = parser.parse_args()

print("Chargement du modèle Whisper...")
model = Whisper(model_path=MODEL_PATH, n_threads=4)

final_text = ""
audio_buffer = np.array([], dtype=np.float32)

# Variables to track our voice activity
has_spoken = False
silent_frames = 0
max_silent_frames = int(args.samplerate * SILENCE_DURATION)

try:
    with sd.InputStream(samplerate=args.samplerate, blocksize=4000, device=DEVICE_ID,
                        dtype="float32", channels=1, callback=callback):
        
        print("#" * 80)
        print(f"Transcription active sur l'appareil {DEVICE_ID}.")
        print(f"Parlez, puis faites silence pendant {SILENCE_DURATION}s pour terminer.")
        print("#" * 80)

        while True:
            data = q.get()
            chunk = data[:, 0]
            audio_buffer = np.concatenate((audio_buffer, chunk))

            # Calculate the volume of this specific fraction of a second
            volume = np.sqrt(np.mean(chunk**2))

            if volume > SILENCE_THRESHOLD:
                if not has_spoken:
                    print("[Système] Voix détectée. Enregistrement en cours...")
                has_spoken = True
                silent_frames = 0  # Reset the silence timer because we are speaking
                
            elif has_spoken:
                # If we have started speaking, but now it's quiet, start the timer
                silent_frames += len(chunk)

            # If the silence timer exceeds our 1.5 second limit, stop!
            if has_spoken and silent_frames >= max_silent_frames:
                print(f"\n[Système] {SILENCE_DURATION}s de silence détectées. Fin de l'enregistrement...")
                break

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Error: {e}")

# --- POST-PROCESSING & TRANSCRIPTION ---
if len(audio_buffer) > 0 and has_spoken:
    print("\nTranscription de l'audio en cours...")
    temp_wav = "/tmp/whisper_chunk.wav"
    sf.write(temp_wav, audio_buffer, args.samplerate)
    
    result = model.transcribe(temp_wav, language="fr")
    final_text = result.get('text', '').strip()
    
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

if final_text:
    print(f"\nTexte brut entendu : {final_text}")
    
    clean_text = re.sub(r'\[.*?\]|\(.*?\)', '', final_text)
    
    clean_text = unidecode(clean_text)
    
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    
    clean_text = clean_text.strip().upper()
    
    if clean_text:
        print(f"Envoi vers alphabet : {clean_text}")
        alphabet.spell(clean_text)
    else:
        print("Audio ignoré (uniquement du bruit de fond ou des balises).")
else:
    print("Aucun texte compris.")
