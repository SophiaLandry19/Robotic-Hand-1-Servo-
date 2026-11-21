"""Webcam hand tracking -> Arduino, two servos.

    Servo 1 (MG996R, pin 9):     pulls all 4 finger strings at once
    Servo 2 (mini servo, pin 10): swings the opposable thumb

Watches your hand through the webcam and measures how bent your fingers
and thumb are (0 = straight, 100 = fully curled). Sends two numbers to the
Arduino: fingers, thumb.

Run:   press the Run button in VS Code
Keys (click the camera window first):
    o = save your OPEN hand as 0%   (fingers straight, thumb out)
    f = save your FIST as 100%      (fingers curled, thumb folded in)
    r = reset to the built-in settings
    t = servo setup: set where the robot fingers and thumb are straight
        and curled (1-2 pick a servo, - and + move it, s = save straight,
        c = save curled, t again = done)
    q = quit

The first run installs the packages it needs by itself.
"""

import importlib
import json
import math
import os
import site
import subprocess
import sys
import time
import urllib.request
from collections import deque

# ---------- installs anything that is missing (first run only) ----------
NEEDED = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "serial": "pyserial",
    "certifi": "certifi",
}


def install_missing():
    missing = []
    for module, package in NEEDED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if not missing:
        return
    print("Installing:", ", ".join(missing), "- one time only")
    pip = [sys.executable, "-m", "pip", "install"]
    tries = ([], ["--user"], ["--user", "--break-system-packages"])
    for extra in tries:
        if subprocess.call(pip + extra + missing) == 0:
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.append(user_site)
            importlib.invalidate_caches()
            return
    raise SystemExit("Could not install the packages. Copy the error "
                     "text above and ask for help.")


install_missing()
# ------------------------------------------------------------------------

import ssl

import certifi
import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- settings you may want to change ----------------
PORT = None        # None = find the Arduino automatically.
                   # Or type it in, e.g. "COM5" or "/dev/cu.usbserial-10"
BAUD = 115200      # must match Serial.begin() in the Arduino sketch
CAMERA = 0         # 0 = built-in webcam, 1 = second camera
SMOOTHING = 0.3    # 0.1 = smooth and slow, 1.0 = instant and jittery
MARGIN = 0.1       # dead zone at each end after calibrating (0 to 0.3)

# Setup mode can move a servo a little past 0 and 180 degrees, to reach
# its real end stops. Most servos stop moving around -10 to -15 degrees;
# pressing further only makes them buzz.
LOWEST = -14       # lowest angle setup mode will go to (degrees)
HIGHEST = 194      # highest angle setup mode will go to (degrees)
STEP = 2           # setup mode: degrees per key press

# The robot hand only starts copying you after it sees an open palm.
OPEN_TRIGGER = 25  # fingers and thumb must read at or below this percent
HOLD_TIME = 1.0    # seconds the open palm must be held to start
PAUSE_AFTER = 2.0  # seconds with no hand in view before it pauses again

# The MG996R gets hot if it squeezes at full strength for a long time.
GRIP_LIMIT = 8.0   # seconds at a full fist before the fingers ease off
GRIP_FULL = 90     # percent that counts as a full fist
GRIP_EASE = 70     # percent the fingers ease off to
# -----------------------------------------------------------------

# Built-in starting points, used until you press o and f.
# Order: thumb, index, middle, ring, pinky (degrees of bend), then
# thumb reach (thumb tip to pinky knuckle, in palm widths).
DEFAULT_OPEN = [20, 30, 30, 30, 30, 1.7]
DEFAULT_FIST = [90, 200, 200, 200, 200, 0.7]
MIN_SPAN = [15, 40, 40, 40, 40, 0.2]   # smallest believable difference

# Where each robot servo is straight and curled, in servo degrees.
# Order: fingers (MG996R), thumb (mini servo).
# Set these with setup mode (press t); they can go below 0.
DEFAULT_STRAIGHT = [0, 0]
DEFAULT_CURLED = [150, 90]

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "hand_landmarker.task")
SAVE_FILE = os.path.join(HERE, "hand_calibration.json")
POSITIONS_FILE = os.path.join(HERE, "servo_angles.json")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/"
             "hand_landmarker.task")

# Landmark numbers along each finger, from the wrist to the fingertip.
CHAINS = [
    [1, 2, 3, 4],          # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]
NAMES = ["Fingers", "Thumb"]
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def find_arduino():
    """Return the serial port name of the Arduino, or None."""
    if PORT:
        return PORT
    words = ("arduino", "ch340", "wch", "usbserial", "usbmodem",
             "usb serial", "usb-serial", "ttyusb", "ttyacm")
    for p in serial.tools.list_ports.comports():
        text = f"{p.device} {p.description} {p.manufacturer}".lower()
        if any(w in text for w in words):
            return p.device
    return None


def angle_between(a, b):
    """Angle in degrees between two 3D directions."""
    dot = sum(x * y for x, y in zip(a, b))
    len_a = math.sqrt(sum(x * x for x in a))
    len_b = math.sqrt(sum(x * x for x in b))
    if len_a == 0 or len_b == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (len_a * len_b)))
    return math.degrees(math.acos(cosine))


def total_bend(points, chain):
    """Add up the bend at every joint along one finger, in degrees."""
    bones = []
    for a, b in zip(chain, chain[1:]):
        bones.append(tuple(q - p for p, q in zip(points[a], points[b])))
    return sum(angle_between(u, v) for u, v in zip(bones, bones[1:]))


def raw_values(points):
    """points = 21 (x, y, z) landmarks. Returns six raw measurements:
    bend of thumb, index, middle, ring, pinky, then thumb reach."""
    raw = [total_bend(points, chain) for chain in CHAINS]
    palm_width = math.dist(points[5], points[17])
    if palm_width:
        raw.append(math.dist(points[4], points[17]) / palm_width)
    else:
        raw.append(DEFAULT_OPEN[5])
    return raw


def to_percent(value, at_zero, at_hundred):
    span = at_hundred - at_zero
    if span == 0:
        return 0.0
    return max(0.0, min(1.0, (value - at_zero) / span)) * 100


def percents(raw, lo, hi):
    """Turn six raw measurements into the two servo values, 0-100.
    Fingers = average of the 4 fingers (one servo pulls them all).
    Thumb = average of how bent the thumb is and how far it reaches
    across the palm (the mini servo swings the whole thumb)."""
    p = [to_percent(raw[i], lo[i], hi[i]) for i in range(6)]
    fingers = (p[1] + p[2] + p[3] + p[4]) / 4
    thumb = (p[0] + p[5]) / 2
    return [fingers, thumb]


def calibrate(which, raw, lo, hi):
    """Save the current hand as the open (0%) or fist (100%) position.
    Returns how many of the six measurements were accepted."""
    accepted = 0
    for i in range(6):
        if which == "open":
            span = hi[i] - raw[i]
            new_lo, new_hi = raw[i] + MARGIN * span, hi[i]
        else:
            span = raw[i] - lo[i]
            new_lo, new_hi = lo[i], raw[i] - MARGIN * span
        direction = DEFAULT_FIST[i] - DEFAULT_OPEN[i]
        right_way = (new_hi - new_lo) * direction > 0
        if right_way and abs(new_hi - new_lo) >= MIN_SPAN[i]:
            lo[i], hi[i] = new_lo, new_hi
            accepted += 1
    return accepted


def update_trigger(state, seen, targets, now):
    """Decide whether the robot hand should be copying you right now.
    state = [mirroring, open_since, last_seen]; it is changed in place."""
    mirroring, open_since, last_seen = state
    if seen:
        last_seen = now
    if not mirroring:
        is_open = seen and max(targets) <= OPEN_TRIGGER
        if not is_open:
            open_since = None
        elif open_since is None:
            open_since = now
        elif now - open_since >= HOLD_TIME:
            mirroring = True
    elif now - last_seen > PAUSE_AFTER:
        mirroring = False
        open_since = None
    state[0], state[1], state[2] = mirroring, open_since, last_seen
    return mirroring


def limit_grip(fingers, grip, now):
    """Ease the fingers off after GRIP_LIMIT seconds at a full fist, so the
    MG996R does not overheat. grip = [full_since, eased]; changed in place.
    The fingers squeeze fully again once you open your hand a bit."""
    full_since, eased = grip
    if fingers >= GRIP_FULL:
        if full_since is None:
            full_since = now
        elif now - full_since >= GRIP_LIMIT:
            eased = True
    else:
        full_since = None
    if eased and fingers < GRIP_EASE - 20:
        eased = False
    grip[0], grip[1] = full_since, eased
    return min(fingers, GRIP_EASE) if eased else fingers


def to_servo(values, straight, curled):
    """Turn hand percents (0 = open) into servo angles in degrees,
    so that an open hand puts the robot fingers at their straight spot."""
    out = []
    for i, v in enumerate(values):
        out.append(straight[i] + (curled[i] - straight[i]) * v / 100)
    return out


def to_pulse(angle):
    """Servo angle in degrees -> pulse width in microseconds.
    Uses the same scale as the Arduino Servo library (0 deg = 544,
    180 deg = 2400), extended a little past both ends."""
    angle = max(LOWEST, min(HIGHEST, angle))
    return round(544 + angle * (2400 - 544) / 180)


def send(arduino, angles):
    if arduino:
        pulses = [to_pulse(a) for a in angles]
        line = ",".join(str(p) for p in pulses) + "\n"
        arduino.write(line.encode())


def load_positions():
    try:
        with open(POSITIONS_FILE) as f:
            saved = json.load(f)
        straight = [int(n) for n in saved["straight"]]
        curled = [int(n) for n in saved["curled"]]
        if len(straight) == 2 and len(curled) == 2:
            return straight, curled
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return list(DEFAULT_STRAIGHT), list(DEFAULT_CURLED)


def save_positions(straight, curled):
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump({"straight": straight, "curled": curled}, f)
    except OSError:
        pass


def load_calibration():
    try:
        with open(SAVE_FILE) as f:
            saved = json.load(f)
        lo, hi = list(saved["lo"]), list(saved["hi"])
        if len(lo) == 6 and len(hi) == 6:
            return lo, hi
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return list(DEFAULT_OPEN), list(DEFAULT_FIST)


def save_calibration(lo, hi):
    try:
        with open(SAVE_FILE, "w") as f:
            json.dump({"lo": lo, "hi": hi}, f)
    except OSError:
        pass


def download_model():
    print("Downloading the hand-tracking model (one time only)...")
    secure = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=secure) as reply:
        data = reply.read()
    with open(MODEL_FILE, "wb") as f:
        f.write(data)


def main():
    if not os.path.exists(MODEL_FILE):
        download_model()

    port = find_arduino()
    arduino = None
    if port:
        try:
            arduino = serial.Serial(port, BAUD, timeout=1)
            time.sleep(2)  # the Arduino restarts when the port opens
            print(f"Connected to the Arduino on {port}")
        except serial.SerialException as error:
            print(f"Could not open {port}: {error}")
            print("Close the Arduino IDE Serial Monitor and run again.")
            arduino = None
    if not arduino:
        print("No Arduino connected - running camera only (preview mode).")

    base = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    camera = cv2.VideoCapture(CAMERA)
    if not camera.isOpened():
        raise SystemExit("Could not open the camera. "
                         "Check camera permission.")

    lo, hi = load_calibration()
    straight, curled = load_positions()
    setup = False         # setup mode: move the servos by keyboard
    pos = list(straight)  # where each servo is during setup mode
    pick = 0              # servo being adjusted (0 = fingers, 1 = thumb)
    recent = deque(maxlen=10)   # last few raw readings, for calibrating
    smooth = [0.0] * 2
    last_sent = [-10] * 2
    last_time = 0.0
    misses = 0
    trigger = [False, None, 0.0]  # mirroring, open_since, last_seen
    grip = [None, False]          # full_since, eased
    targets = [0.0] * 2
    note = ""
    note_until = 0.0
    start = time.monotonic()

    with vision.HandLandmarker.create_from_options(options) as tracker:
        while True:
            ok, frame = camera.read()
            if not ok:
                misses += 1
                if misses > 100:  # about 5 seconds with no picture
                    print("The camera sent no picture. Quit VS Code, "
                          "reopen it and run again. If that fails, "
                          "change CAMERA = 0 to CAMERA = 1.")
                    break
                time.sleep(0.05)
                continue
            misses = 0
            frame = cv2.flip(frame, 1)  # mirror view
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            stamp = int((time.monotonic() - start) * 1000)
            result = tracker.detect_for_video(image, stamp)

            seen = bool(result.hand_landmarks
                        and result.hand_world_landmarks)
            if seen:
                marks = result.hand_landmarks[0]
                marks3d = result.hand_world_landmarks[0]
                world = [(lm.x, lm.y, lm.z) for lm in marks3d]
                raw = raw_values(world)
                recent.append(raw)
                targets = percents(raw, lo, hi)
                for i in range(2):
                    smooth[i] += SMOOTHING * (targets[i] - smooth[i])
                pixels = [(int(lm.x * w), int(lm.y * h)) for lm in marks]
                for a, b in BONES:
                    cv2.line(frame, pixels[a], pixels[b], WHITE, 2)
                for x, y in pixels:
                    cv2.circle(frame, (x, y), 4, GREEN, -1)
            else:
                recent.clear()  # the servos hold their last position

            now = time.monotonic()
            values = [round(v) for v in smooth]
            values[0] = round(limit_grip(values[0], grip, now))
            mirroring = update_trigger(trigger, seen, targets, now)
            changed = any(abs(v - s) >= 2
                          for v, s in zip(values, last_sent))
            ready = seen and changed and now - last_time >= 0.05
            if mirroring and ready and not setup:  # after an open palm
                send(arduino, to_servo(values, straight, curled))
                last_sent = values
                last_time = now

            if seen:
                for i, name in enumerate(NAMES):
                    text = f"{name}: {values[i]}%"
                    if i == 0 and grip[1]:
                        text += "  (easing off to rest the servo)"
                    cv2.putText(frame, text, (20, 40 + 32 * i),
                                FONT, 0.8, GREEN, 2)
            else:
                cv2.putText(frame, "No hand seen", (20, 40),
                            FONT, 0.8, GREEN, 2)
            if setup:
                status = (f"SETUP - {NAMES[pick]} at {pos[pick]} deg  "
                          f"(straight {straight[pick]}, "
                          f"curled {curled[pick]})")
            elif mirroring:
                status = "MIRRORING - the robot hand is copying you"
            elif trigger[1] is not None:
                status = "Open palm seen - hold still..."
            else:
                status = "PAUSED - show an open palm to start"
            cv2.putText(frame, status, (20, 40 + 32 * 2 + 10),
                        FONT, 0.8, WHITE, 2)
            if setup:
                keys_help = ("1 = fingers  2 = thumb   - / + move   "
                             "s = straight   c = curled   t = done")
            else:
                keys_help = ("o = open hand   f = fist   r = reset   "
                             "t = servo setup   q = quit")
            cv2.putText(frame, keys_help, (20, h - 20),
                        FONT, 0.6, WHITE, 2)
            if now < note_until:
                cv2.putText(frame, note, (20, h - 55), FONT, 0.8, GREEN, 2)

            cv2.imshow("Hand tracking - press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                setup = not setup
                if setup:
                    pos = list(straight)
                    send(arduino, pos)  # both servos to their straight spot
                    note = "Setup: 1 = fingers, 2 = thumb"
                else:
                    trigger[:] = [False, None, now]
                    last_sent = [-10] * 2
                    note = "Setup done - show an open palm to start"
                note_until = now + 3
            elif setup:
                if key in (ord("1"), ord("2")):
                    pick = key - ord("1")
                elif key in (ord("-"), ord("_")):
                    pos[pick] = max(LOWEST, pos[pick] - STEP)
                    send(arduino, pos)
                elif key in (ord("="), ord("+")):
                    pos[pick] = min(HIGHEST, pos[pick] + STEP)
                    send(arduino, pos)
                elif key in (ord("s"), ord("c")):
                    if key == ord("s"):
                        straight[pick] = pos[pick]
                        word = "straight"
                    else:
                        curled[pick] = pos[pick]
                        word = "curled"
                    save_positions(straight, curled)
                    note = f"{NAMES[pick]} {word} position saved"
                    if straight[pick] == curled[pick]:
                        note = (f"{NAMES[pick]}: straight and curled "
                                f"are the same - move it first")
                    note_until = now + 3
            elif key in (ord("o"), ord("f")):
                if len(recent) < 5:
                    note = "Hold your hand steady in view, then try again"
                else:
                    average = [sum(r[i] for r in recent) / len(recent)
                               for i in range(6)]
                    which = "open" if key == ord("o") else "fist"
                    count = calibrate(which, average, lo, hi)
                    save_calibration(lo, hi)
                    label = "Open hand" if which == "open" else "Fist"
                    note = f"{label} saved ({count} of 6 readings used)"
                note_until = now + 3
            elif key == ord("r"):
                lo, hi = list(DEFAULT_OPEN), list(DEFAULT_FIST)
                try:
                    os.remove(SAVE_FILE)
                except OSError:
                    pass
                note = "Reset to the built-in settings"
                note_until = now + 3

    if arduino:
        arduino.write(b"x\n")  # switch the servos off; nothing moves
        time.sleep(0.2)
        arduino.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
