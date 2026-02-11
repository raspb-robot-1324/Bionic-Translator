#!/usr/bin/env python3

import argparse
import queue
import sys
import json
import sounddevice as sd
from unidecode import unidecode 
from vosk import Model, KaldiRecognizer

# Import your local file
import alphabet 

q = queue.Queue()

# --- CONFIGURATION ---
# Since you are using a French model, the keyword is likely "arrêt" or "stop"
STOP_KEYWORD = "bonjour" 
# ---------------------

def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-l", "--list-devices", action="store_true", help="show list of audio devices and exit")
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[parser])
parser.add_argument("-f", "--filename", type=str, metavar="FILENAME", help="audio file to store recording to")
parser.add_argument("-d", "--device", type=int_or_str, help="input device (numeric ID or substring)")
parser.add_argument("-r", "--samplerate", type=int, help="sampling rate")
parser.add_argument("-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is fr")
args = parser.parse_args(remaining)

# --- MAIN LOGIC ---

# 1. Setup Audio Device & Model (Do this ONCE, outside the loop)
if args.samplerate is None:
    device_info = sd.query_devices(args.device, "input")
    args.samplerate = int(device_info["default_samplerate"])

if args.model is None:
    print("Loading default French model...")
    model = Model(lang="fr")
else:
    model = Model(lang=args.model)

rec = KaldiRecognizer(model, args.samplerate)
final_text = ""

# 2. Open Microphone Stream
try:
    with sd.RawInputStream(samplerate=args.samplerate, blocksize=8000, device=args.device,
            dtype="int16", channels=1, callback=callback):
        
        print("#" * 80)
        print(f"Transcription active. Dites '{STOP_KEYWORD}' pour terminer.")
        print("#" * 80)

        # 3. Audio Processing Loop
        while True:
            data = q.get()
            
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = res.get('text', '')
                
                if text:
                    print(f"Entendu: {text}")
                    # Check for keyword
                    if STOP_KEYWORD in text.lower() or "arrêt" in text.lower():
                        print("\n[Système] Mot-clé détecté. Arrêt...")
                        final_text = text
                        break # <--- IMPORTANT: This breaks the loop to stop recording

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Error: {e}")

# 4. Post-Processing (Happens after the loop breaks)
if final_text:
    print(f"\nTraitement du texte: {final_text}")
    
    # Use unidecode (not unicode) to remove accents
    clean_text = unidecode(final_text)
    
    # Remove the stop word so it doesn't get spelled out
    clean_text = clean_text.replace(STOP_KEYWORD, "").replace("arret", "").strip().upper()
    
    print(f"Envoi vers alphabet: {clean_text}")
    
    # Call your other file
    alphabet.spell(clean_text)
else:
    print("Aucun texte enregistré.")
