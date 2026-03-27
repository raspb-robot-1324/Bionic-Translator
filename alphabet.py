from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

# Servo definitions
ring = kit.servo[10]
pinky = kit.servo[11]
middle = kit.servo[12]
index = kit.servo[13]
thumb = kit.servo[14]
wrist = kit.servo[15]

# Setup pulse widths
pinky.set_pulse_width_range(500, 2500)
index.set_pulse_width_range(500, 2500)
ring.set_pulse_width_range(500, 2500)
middle.set_pulse_width_range(500, 2500)
thumb.set_pulse_width_range(500, 2500)
wrist.set_pulse_width_range(500, 2500)

# Setup actuation ranges
pinky.actuation_range = 270
ring.actuation_range = 270
middle.actuation_range = 270
index.actuation_range = 270
thumb.actuation_range = 270
wrist.actuation_range = 270

all_fingers = [pinky, ring, middle]

def reset():
    time.sleep(1)
    pinky.angle = 0
    ring.angle = 0
    middle.angle = 0
    index.angle = 0
    thumb.angle = 0
    wrist.angle = 135
    time.sleep(1)
    

def A():
    index.angle = 110
    for i in all_fingers:
        i.angle = 110

def B():
    thumb.angle = 110

def C():
    wrist.angle = 210
    A()
    thumb.angle = 110

def D():
    wrist.angle = 210
    thumb.angle = 110
    for i in all_fingers:
        i.angle = 110

def E():
    index.angle = 110
    for i in all_fingers:
        i.angle = 110
    thumb.angle = 110

def F():
    wrist.angle = 40
    thumb.angle = 110
    index.angle = 110

def G():
    wrist.angle = 210
    time.sleep(0.5)
    thumb.angle = 100
    for i in all_fingers:
        i.angle = 110
    index.angle = 60

def H():
    wrist.angle = 210
    time.sleep(0.5)
    thumb.angle = 100
    pinky.angle = 140
    ring.angle = 140
    middle.angle = 100
    index.angle = 60


def I():
    ring.angle = 140
    middle.angle = 140
    index.angle = 110
    thumb.angle = 110

def J():
    wrist.angle = 40
    time.sleep(0.5)
    I()

def K():
    wrist.angle = 210
    time.sleep(0.5)
    pinky.angle = 140
    ring.angle = 140
    thumb.angle = 110

def L():
    for i in all_fingers:
        i.angle = 110

def M():
    pinky.angle = 140
    ring.angle = 100
    middle.angle = 100
    index.angle = 70
    thumb.angle = 100

def N():
    pinky.angle = 140
    ring.angle = 140
    middle.angle = 100
    index.angle = 70
    thumb.angle = 100

def O():
    wrist.angle = 210
    time.sleep(0.5)
    thumb.angle = 110
    index.angle = 120
    for i in all_fingers:
        i.angle = 110

def P():
    wrist.angle = 210
    time.sleep(0.5)
    thumb.angle = 50
    index.angle = 70
    middle.angle = 100
    ring.angle = 140
    pinky.angle = 140

def Q():
    wrist.angle = 210
    time.sleep(0.5)
    thumb.angle = 100
    pinky.angle = 140
    ring.angle = 120
    middle.angle = 110
    index.angle = 80

def R():
    pinky.angle = 110
    ring.angle = 110

def S():
    wrist.angle = 210
    time.sleep(1)
    index.angle = 100
    for i in all_fingers:
        i.angle = 110
    thumb.angle = 100

def T():
    wrist.angle = 40
    time.sleep(1)
    index.angle = 100
    for i in all_fingers:
        i.angle = 110
    thumb.angle = 110

def U():
    pinky.angle = 110
    ring.angle = 110
    thumb.angle = 110

def V():
    wrist.angle = 40
    time.sleep(0.5)
    pinky.angle = 140
    ring.angle = 140
    thumb.angle = 110

def W():
    pinky.angle = 140
    thumb.angle = 110

def X():
    for i in all_fingers:
        i.angle = 110
    index.angle = 70
    thumb.angle = 110

def Y():
    pinky.angle = 80
    ring.angle = 140
    middle.angle = 140
    index.angle = 110

def Z():
    wrist.angle = 40
    time.sleep(0.5)
    for i in all_fingers:
        i.angle = 100
    index.angle = 100
    thumb.angle = 100

def spell(word):
    # Ensure word is uppercase so it matches function names A(), B(), etc.
    word = word.upper()
    for char in word:
        if char == " ":
            reset()
            time.sleep(0.5)
        elif 'A' <= char <= 'Z': # Only execute if it's a letter
            reset()
            # Dynamically call the function matching the letter
            globals()[char]()
            time.sleep(0.5)
	    
	    
    reset()
    
reset()
# ~ spell("Bonjour")

# ~ reset()
# ~ spell("Z")
# ~ wrist.angle = 135
# ~ spell("Hello Dad")
# ~ wrist.angle = 210
# ~ time.sleep(0.5)
# ~ wrist.angle = 135
# Initialize hand position
# ~ reset()
# ~ D()
# ~ reset()
# ~ Y()
# ~ reset()
# ~ L()
# ~ reset()
# ~ A()
# ~ reset()
# ~ N()

# ~ reset()

