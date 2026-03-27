import socket
import alphabet

# --- Robot Hand Integration ---
# ~ try:
    # ~ from robot_hand.alphabet import spell
    # ~ print("? Successfully connected to robot_hand.alphabet")
# ~ except ImportError:
    # ~ print("?? Warning: robot_hand/alphabet.py not found. Using mock spell() for testing.")
    # ~ def spell(word):
        # ~ print(f"? [ROBOT HAND MOCK EXECUTION] Spelling: {word}")

def start_server(host='0.0.0.0', port=65432):
    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"\n? Pi Server is actively listening on port {port}...")
        print("Waiting for your Mac to send a word...\n")
        
        while True:
            # Wait for a connection from the Mac
            conn, addr = s.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    # Decode the received bytes into a string
                    word = data.decode('utf-8').strip()
                    print(f"? Received from Mac: '{word}'")
                    
                    # Trigger the robot hand!
                    alphabet.spell(word)

if __name__ == "__main__":
    start_server()
