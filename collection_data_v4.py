"""GestureFlow V4: webcam and imported-video gesture dataset collection.

Uses the exact V3 108-column CSV feature schema (Timestamp + 107 fields),
with stricter sampling, per-alphabet sample caps, videos and snapshots.
"""

import csv
import re
import shutil
from pathlib import Path

import cv2
import math
import time
import os
from collections import deque
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------- SETTINGS --------------------
# Both axes must move by at least this normalized fraction of frame size,
# measured against the last ACCEPTED sample (or initial valid pose).
# At 640x480: 0.025 corresponds to about 16px horizontally and 12px vertically.
X_VARIATION_THRESHOLD = 0.025
Y_VARIATION_THRESHOLD = 0.025
MAX_ROWS = 100
SNAPSHOT_EVERY = 1          # exactly one image per accepted sample
START_DELAY_SECONDS = 3  # only for live camera; recorded video is processed from frame 1
DUPLICATE_RMS_THRESHOLD = 0.012  # normalized XY landmark RMS vs ANY saved sample
VIDEO_FPS_FALLBACK = 30.0
BASE_FOLDER = Path(__file__).resolve().parent

# -------------------- HAND FUNCTIONS --------------------
def calculate_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)   

def calculate_angle(a, b, c):   
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)

    dot = sum(x * y for x, y in zip(v1, v2))
    magnitude = math.sqrt(sum(x ** 2 for x in v1)) * math.sqrt(sum(x ** 2 for x in v2))

    if magnitude == 0:
        return 0

    return math.degrees(math.acos(max(-1, min(1, dot / magnitude))))

def get_hand_orientation(hand, label):
    p = hand.landmark

    wrist = p[0]
    index = p[5]
    pinky = p[17]

    # Calculate the orientation of the palm using the hand landmark plane.
    cross_z = (
        (index.x - wrist.x) * (pinky.y - wrist.y)
        - (index.y - wrist.y) * (pinky.x - wrist.x)
    )

    sign = 1 if label == "Right" else -1
    oriented_cross = cross_z * sign

    # Front / Back threshold.
    # If the palm is almost parallel to the camera, it is treated as Side.
    SIDE_THRESHOLD = 0.015

    if abs(oriented_cross) < SIDE_THRESHOLD:
        return "Side"

    if oriented_cross > 0:
        return "Front"

    return "Back"

# -------------------- FINGER STATE DETECTION --------------------
# Finger encoding: 0 = Closed, 1 = Open.
#
# The old rule measured tip-to-knuckle distance in 2D (x, y only) and divided
# by palm width. That breaks as soon as the hand turns: a finger curled toward
# the camera looks short, and a bent finger seen from the side looks long, so
# closed fingers were often stored as open. The rule below uses the 3D world
# landmarks (x, y AND z), so the result does not depend on how the hand is
# rotated.
#
#   Index / Middle / Ring / Little
#     reach     = |wrist -> tip| / |wrist -> MCP knuckle|
#                 (about 1.6-2.2 when straight, about 0.6-0.9 when curled)
#     pip_angle = angle at the middle joint (MCP-PIP-DIP), about 160-180 when
#                 straight
#       reach >= FINGER_OPEN_REACH    -> open
#       reach <= FINGER_CLOSED_REACH  -> closed
#       in between (half bent)        -> the PIP angle decides
#
#   Thumb
#     open only if the tip joint is reasonably straight (IP angle) AND the tip
#     is held away from the hand. A closed thumb rests on the curled fingers or
#     on the palm, so its tip stays close to the index / middle / ring finger
#     joints (distance in palm widths). A thumb pressed flat against the side of
#     the hand therefore counts as closed; set THUMB_AWAY_FROM_HAND to 0 if you
#     want any straight thumb to count as open.
#
# If a finger is still read wrongly, press "d" in the camera window to see
# the stored states live, then adjust the thresholds below.
FINGER_OPEN_REACH = 1.45
FINGER_CLOSED_REACH = 1.15
FINGER_PIP_OPEN_ANGLE = 135        # degrees
THUMB_IP_OPEN_ANGLE = 130          # degrees
THUMB_AWAY_FROM_HAND = 0.55        # palm widths
THUMB_NEAR_JOINTS = (5, 6, 7, 9, 10, 11, 13, 14)   # index / middle / ring joints
SMOOTH_FRAMES = 5                  # majority vote over the last N frames

FINGER_JOINTS = {                  # (MCP, PIP, DIP, TIP) landmark indexes
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Little": (17, 18, 19, 20)
}

def distance_3d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def angle_3d(a, b, c):
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)

    dot = sum(x * y for x, y in zip(v1, v2))
    magnitude = math.sqrt(sum(x ** 2 for x in v1)) * math.sqrt(sum(x ** 2 for x in v2))

    if magnitude == 0:
        return 0

    return math.degrees(math.acos(max(-1, min(1, dot / magnitude))))

def classify_fingers(world):
    # Single-frame classification. Returns {"Thumb": 0/1, ...} or None if the
    # hand geometry is unusable.
    p = world.landmark

    palm_width = distance_3d(p[5], p[17])

    if palm_width <= 0:
        return None

    status = {}

    # Thumb
    ip_angle = angle_3d(p[2], p[3], p[4])
    away_from_hand = min(distance_3d(p[4], p[i]) for i in THUMB_NEAR_JOINTS) / palm_width

    if ip_angle >= THUMB_IP_OPEN_ANGLE and away_from_hand >= THUMB_AWAY_FROM_HAND:
        status["Thumb"] = 1
    else:
        status["Thumb"] = 0

    # Index, Middle, Ring, Little
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        wrist_to_mcp = distance_3d(p[0], p[mcp])

        if wrist_to_mcp <= 0:
            return None

        reach = distance_3d(p[0], p[tip]) / wrist_to_mcp

        if reach >= FINGER_OPEN_REACH:
            status[name] = 1
        elif reach <= FINGER_CLOSED_REACH:
            status[name] = 0
        else:
            pip_angle = angle_3d(p[mcp], p[pip], p[dip])
            status[name] = 1 if pip_angle >= FINGER_PIP_OPEN_ANGLE else 0

    return status

def get_hand_position(status):
    # Full hand position: 0 = every finger closed, 1 = every finger open,
    # 2 = mixed.
    states = list(status.values())

    if all(state == 0 for state in states):
        return 0

    if all(state == 1 for state in states):
        return 1

    return 2

def get_finger_status(world):
    # Stateless version (no smoothing).
    status = classify_fingers(world)

    if status is None:
        return None, None

    return status, get_hand_position(status)

class FingerTracker:
    # One tracker per hand (Left / Right). Keeps the last few frames and
    # returns the majority vote for each finger, so a single noisy frame
    # cannot flip a stored finger state.

    def __init__(self, window=SMOOTH_FRAMES):
        self.history = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def update(self, world):
        raw = classify_fingers(world)

        if raw is None:
            return None, None

        self.history.append(raw)
        frames = len(self.history)

        status = {}

        for name in raw:
            open_votes = sum(frame_state[name] for frame_state in self.history)

            if open_votes * 2 > frames:
                status[name] = 1
            elif open_votes * 2 < frames:
                status[name] = 0
            else:
                status[name] = raw[name]     # tie: trust the newest frame

        return status, get_hand_position(status)

# -------------------- DISPLAY FUNCTIONS --------------------
def draw_landmark_numbers(frame, hand):
    h, w = frame.shape[:2]

    for i, point in enumerate(hand.landmark):
        x = int(point.x * w)
        y = int(point.y * h)

        cv2.putText(frame,str(i),(x + 4, y - 4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0, 0, 0),3,cv2.LINE_AA)    # Black outline
        cv2.putText(frame,str(i),(x + 4, y - 4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255, 255, 0),1,cv2.LINE_AA)  # Landmark number

def draw_text(frame,text,y,color=(45, 45, 45),outline=(235, 235, 235),thickness=1):

    cv2.putText(frame,text,(28, y),cv2.FONT_HERSHEY_SIMPLEX,0.72,outline,thickness + 3,cv2.LINE_AA)
    cv2.putText(frame,text,(28, y),cv2.FONT_HERSHEY_SIMPLEX,0.72,color,thickness,cv2.LINE_AA)

def save_snapshot(clean_frame, all_hands, path):
    # clean_frame is the camera image with no text on it. The hand landmarks
    # are drawn only now, when a snapshot is actually due, so normal frames
    # do not pay for the extra drawing.
    for hand in all_hands:
        mp_drawing.draw_landmarks(
            clean_frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )
        draw_landmark_numbers(clean_frame, hand)

    return cv2.imwrite(path, clean_frame)

# -------------------- CSV COLUMNS --------------------
TWO_HAND_ALPHABETS = (
    "A", "B", "D", "E", "F", "G", "H", "J", "K",
    "M", "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"
)

ONE_HAND_ALPHABETS = ("C", "O", "I", "L", "U", "V")

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]

# Row layout (index 0 = first value AFTER the Timestamp column).
# EVERY alphabet now uses this same layout, so all CSV files have
# identical columns.
#
#   Left coordinates      : 0-41
#   Left fingers          : 42-46
#   Left orientation      : 47
#   Left hand position    : 48
#   Right coordinates     : 49-90
#   Right fingers         : 91-95
#   Right orientation     : 96
#   Right hand position   : 97
#   Wrist distance X      : 98
#   Wrist distance Y      : 99
#   Left  thumb-index dist (4-8) : 100
#   Left  dist 4-5               : 101
#   Left  dist 4-6               : 102
#   Right thumb-index dist (4-8) : 103
#   Right dist 4-5               : 104
#   Right dist 4-6               : 105
HAND_LAYOUT = {
    "Left":  {"coords": 0,  "fingers": 42, "orientation": 47, "position": 48, "dists": 100},
    "Right": {"coords": 49, "fingers": 91, "orientation": 96, "position": 97, "dists": 103},
}
WRIST_DIST_X_INDEX = 98
WRIST_DIST_Y_INDEX = 99
LEFT_RIGHT_INDEX_DIST_INDEX = 106


def get_csv_columns():
    columns = ["Timestamp"]

    for side, name in (("L", "Left"), ("R", "Right")):
        for i in range(21):
            columns.extend([f"{side}x{i}", f"{side}y{i}"])

        columns.extend([
            f"{side}T", f"{side}I", f"{side}M", f"{side}R", f"{side}L",
            f"{name}_Orientation",
            f"{name}_Hand_Position"
        ])

    # Wrist-to-wrist distance split into its horizontal and vertical parts.
    columns.extend(["Wrist_Dist_X", "Wrist_Dist_Y"])

    # Per-hand distances between landmarks:
    #   Thumb_Index_Dist = thumb tip (4) to index fingertip (8)
    #   Dist_4_5         = thumb tip (4) to index MCP (5)
    #   Dist_4_6         = thumb tip (4) to index PIP (6)
    for name in ("Left", "Right"):
        columns.extend([
            f"{name}_Thumb_Index_Dist",
            f"{name}_Dist_4_5",
            f"{name}_Dist_4_6"
        ])

    # Distance between the left and right index fingertips (landmark 8).
    columns.append("Left_Right_Index_Dist")

    return columns

def create_empty_row(alphabet):
    row_length = len(get_csv_columns()) - 1

    # One-hand alphabets: every column starts at 0, so the hand that is
    # NOT shown automatically keeps all-0 values (coordinates, fingers,
    # orientation, hand position) and the wrist distance stays 0.
    if alphabet in ONE_HAND_ALPHABETS:
        return [0] * row_length

    # Two-hand alphabets: both hands are required, so every value gets
    # overwritten. NA is only a fallback if a hand could not be processed.
    return [pd.NA] * row_length


def add_hand_to_row(row, label, hand, finger_status, hand_position):
    layout = HAND_LAYOUT[label]

    for i, point in enumerate(hand.landmark):
        row[layout["coords"] + i * 2] = point.x
        row[layout["coords"] + i * 2 + 1] = point.y

    for i, name in enumerate(FINGER_NAMES):
        row[layout["fingers"] + i] = finger_status[name]

    orientation = get_hand_orientation(hand, label)
    orientation_code = {"Front": "F", "Back": "B", "Side": "S"}[orientation]
    row[layout["orientation"]] = orientation_code
    row[layout["position"]] = hand_position

    # Landmark distances (MediaPipe normalized image coordinates):
    # thumb tip-index tip (4-8), thumb tip-index MCP (4-5),
    # thumb tip-index PIP (4-6).
    p = hand.landmark
    row[layout["dists"]] = calculate_distance(p[4], p[8])
    row[layout["dists"] + 1] = calculate_distance(p[4], p[5])
    row[layout["dists"] + 2] = calculate_distance(p[4], p[6])


def add_wrist_distance(row, detected_hands):
    # Wrist landmark = index 0.
    # Distance is split into X (horizontal) and Y (vertical) components.
    # Values use MediaPipe's normalized image coordinates (0 to 1).
    # With only one hand visible there is no wrist-to-wrist distance,
    # so both values stay 0.
    if "Left" in detected_hands and "Right" in detected_hands:
        left_wrist = detected_hands["Left"][0].landmark[0]
        right_wrist = detected_hands["Right"][0].landmark[0]

        row[WRIST_DIST_X_INDEX] = abs(left_wrist.x - right_wrist.x)
        row[WRIST_DIST_Y_INDEX] = abs(left_wrist.y - right_wrist.y)
    else:
        row[WRIST_DIST_X_INDEX] = 0
        row[WRIST_DIST_Y_INDEX] = 0


def add_left_right_index_distance(row, detected_hands):
    # Index fingertip = landmark 8 on each hand.
    # Store the 2D Euclidean distance in normalized image coordinates.
    if "Left" in detected_hands and "Right" in detected_hands:
        left_index_tip = detected_hands["Left"][0].landmark[8]
        right_index_tip = detected_hands["Right"][0].landmark[8]
        row[LEFT_RIGHT_INDEX_DIST_INDEX] = calculate_distance(
            left_index_tip, right_index_tip
        )
    else:
        row[LEFT_RIGHT_INDEX_DIST_INDEX] = 0


# -------------------- SAFE FOLDER / CSV HELPERS --------------------

def normalize_username(raw_name):
    """Do not let a user-provided name become a path or an invalid Windows name."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', raw_name.strip()).strip(' .')
    if not name or name.upper() in {
        'CON', 'PRN', 'AUX', 'NUL',
        *(f'COM{i}' for i in range(1, 10)),
        *(f'LPT{i}' for i in range(1, 10)),
    }:
        raise ValueError('Please enter a valid username.')
    return name


def get_paths(username, alphabet):
    """<root>/<user>/<user>_Dataset/A.csv + <user>_Snapshots/A/*.png."""
    user_dir = BASE_FOLDER / username
    dataset_dir = user_dir / f'{username}_Dataset'
    snapshot_dir = user_dir / f'{username}_Snapshots' / alphabet
    return user_dir, dataset_dir, snapshot_dir, dataset_dir / f'{alphabet}.csv'


def ensure_folders(user_dir, dataset_dir, snapshot_dir):
    for folder in (user_dir, dataset_dir, snapshot_dir):
        folder.mkdir(parents=True, exist_ok=True)


def load_existing_data(csv_file):
    """Never replace, rename or silently combine an incompatible CSV."""
    if not csv_file.exists():
        return []
    df = pd.read_csv(csv_file, keep_default_na=True)
    if list(df.columns) != get_csv_columns():
        raise ValueError(
            f'The existing CSV has a different column schema:\n{csv_file}\n'
            'For safety it was NOT modified. Back it up and convert it to '
            'the V3/V4 CSV format before continuing.'
        )
    if len(df) > MAX_ROWS:
        print(f'Warning: this alphabet already has {len(df)} rows (over the V4 cap).')
    return df.values.tolist()


def append_csv_row(csv_file, data_row):
    """Commit each accepted sample immediately, even if Q ends capture early."""
    first_row = not csv_file.exists() or csv_file.stat().st_size == 0
    with csv_file.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if first_row:
            writer.writerow(get_csv_columns())
        writer.writerow(data_row)


def saved_hand_label(existing_rows):
    """For one-hand letters, keep the same left/right hand across sessions."""
    labels = set()
    for row in existing_rows:
        for label, offset in (('Left', 1), ('Right', 50)):
            # These indexes include the Timestamp at position 0.
            coords = row[offset:offset + 42]
            if any(pd.notna(v) and float(v) != 0.0 for v in coords):
                labels.add(label)
    if len(labels) > 1:
        raise ValueError('Existing one-hand CSV mixes left and right hands; inspect it before adding data.')
    return next(iter(labels)) if labels else None


def landmark_vector_from_row(row, alphabet):
    """V3 XY landmarks in deterministic L then R order, including absent-hand zeros."""
    # Timestamp at position 0: left XY values 1..42, right XY values 50..91.
    indexes = list(range(1, 43)) + list(range(50, 92))
    try:
        values = tuple(float(row[index]) for index in indexes)
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def is_near_duplicate(candidate, previous_vectors):
    """Reject near-identical captured landmarks, even if a pose comes back later."""
    if candidate is None:
        return True
    threshold_squared = DUPLICATE_RMS_THRESHOLD ** 2 * len(candidate)
    for previous in previous_vectors:
        if len(previous) != len(candidate):
            continue
        squared = sum((a - b) ** 2 for a, b in zip(candidate, previous))
        if squared < threshold_squared:
            return True
    return False


def significant_wrist_movement(detected_hands, baseline, alphabet):
    """Require BOTH X and Y movement for EACH required hand."""
    required = (next(iter(detected_hands)),) if alphabet in ONE_HAND_ALPHABETS else ('Left', 'Right')
    for side in required:
        if side not in baseline or side not in detected_hands:
            return False
        current = detected_hands[side][0].landmark[0]
        old_x, old_y = baseline[side]
        if (abs(current.x - old_x) < X_VARIATION_THRESHOLD
                or abs(current.y - old_y) < Y_VARIATION_THRESHOLD):
            return False
    return True


def current_wrists(detected_hands):
    return {
        side: (hand.landmark[0].x, hand.landmark[0].y)
        for side, (hand, _world) in detected_hands.items()
    }


def unique_video_path(user_dir, username, alphabet, suffix):
    """Retain existing recordings; use _02, _03... for later sessions."""
    stem = f'{username}_{alphabet}_video'
    candidate = user_dir / f'{stem}{suffix}'
    number = 2
    while candidate.exists():
        candidate = user_dir / f'{stem}_{number:02d}{suffix}'
        number += 1
    return candidate


def archive_old_video(source, user_dir, username, alphabet):
    """Copy (never move/delete) an old recording into its user folder."""
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.parent == user_dir.resolve() and source.name.startswith(f'{username}_{alphabet}_video'):
        return source  # already archived here
    target = unique_video_path(user_dir, username, alphabet, source.suffix.lower())
    shutil.copy2(source, target)
    return target


# -------------------- GUI: CHOOSE INPUT MODE / OLD VIDEO --------------------

def choose_mode(alphabet):
    """A true drop-down on desktops; numbered terminal fallback without Tk."""
    try:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
        root.title(f'GestureFlow - {alphabet} data source')
        root.geometry('370x180')
        root.resizable(False, False)
        selected = {'mode': None}
        tk.Label(root, text=f'Collect alphabet {alphabet}', font=('Arial', 12, 'bold')).pack(pady=(18, 8))
        tk.Label(root, text='Choose a data source:').pack()
        choice = tk.StringVar(value='Live data (camera)')
        dropdown = ttk.Combobox(
            root, textvariable=choice, state='readonly', width=25,
            values=('Live data (camera)', 'Old data (video file)')
        )
        dropdown.pack(pady=8)
        dropdown.current(0)

        def submit():
            selected['mode'] = 'live' if choice.get().startswith('Live') else 'old'
            root.destroy()

        ttk.Button(root, text='Continue', command=submit).pack()
        root.bind('<Return>', lambda _event: submit())
        root.mainloop()
        return selected['mode']  # None means cancelled
    except Exception as exc:
        print(f'Drop-down unavailable ({exc}); using terminal selection.')
        while True:
            value = input('Select 1 = Live data, 2 = Old video (or exit): ').strip().lower()
            if value in ('1', 'live'):
                return 'live'
            if value in ('2', 'old'):
                return 'old'
            if value in ('exit', 'q'):
                return None
            print('Enter 1 or 2.')


def choose_old_video():
    """Open Explorer/Finder via Tk; accept typed paths without a GUI."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askopenfilename(
            parent=root,
            title='Select an old gesture collection video',
            filetypes=[
                ('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v'),
                ('All files', '*.*'),
            ],
        )
        root.destroy()
        return Path(selected) if selected else None
    except Exception as exc:
        print(f'File chooser unavailable ({exc}).')
        path = input('Enter full path to the old video (blank to cancel): ').strip().strip('"').strip("'")
        return Path(path) if path else None


# -------------------- LIVE CAMERA / VIDEO WRITER --------------------

def open_camera():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError('Cannot open camera. Close other camera applications.')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def make_video_writer(user_dir, username, alphabet, frame, camera):
    h, w = frame.shape[:2]
    fps = float(camera.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or not 1 <= fps <= 120:
        fps = VIDEO_FPS_FALLBACK
    for suffix, codec in (('.mp4', 'mp4v'), ('.avi', 'XVID'), ('.avi', 'MJPG')):
        path = unique_video_path(user_dir, username, alphabet, suffix)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h)
        )
        if writer.isOpened():
            return writer, path
        writer.release()
        if path.exists():
            path.unlink()  # cleanup only a failed, newly-created output
    raise RuntimeError('Could not initialize MP4/AVI video recording (no supported codec).')


# -------------------- FRAME PROCESSING --------------------

def process_frame(frame, hands, trackers, alphabet, locked_side):
    """Extract V3 left/right landmarks, 3D finger states and geometric features."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    detected = {}
    states = {}

    if results.multi_hand_landmarks and results.multi_hand_world_landmarks and results.multi_handedness:
        for hand, world, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_hand_world_landmarks,
            results.multi_handedness
        ):
            side = handedness.classification[0].label
            if side not in ('Left', 'Right'):
                continue
            detected[side] = (hand, world)
            states[side] = trackers[side].update(world)

    for side in trackers:
        if side not in detected:
            trackers[side].reset()

    is_one_hand = alphabet in ONE_HAND_ALPHABETS
    valid_count = (len(detected) == 1) if is_one_hand else ('Left' in detected and 'Right' in detected)
    if not valid_count:
        return results, detected, states, None, 'EXACTLY ONE HAND REQUIRED' if is_one_hand else 'BOTH HANDS REQUIRED'
    if is_one_hand and locked_side and locked_side not in detected:
        return results, detected, states, None, f'{locked_side.upper()} HAND REQUIRED'
    if any(states[side][0] is None for side in detected):
        return results, detected, states, None, 'HAND FEATURES NOT READY'

    data_row = create_empty_row(alphabet)
    for side, (hand, _world) in detected.items():
        finger_status, position = states[side]
        add_hand_to_row(data_row, side, hand, finger_status, position)
    add_wrist_distance(data_row, detected)
    add_left_right_index_distance(data_row, detected)
    return results, detected, states, data_row, None


def draw_live_status(frame, alphabet, count, detected, states, message, debug, countdown):
    y = 42
    draw_text(frame, f'Collect Data: {alphabet}', y)
    y += 34
    if countdown > 0:
        draw_text(frame, f'Starting in {countdown:.1f}s...', y, color=(0, 0, 200))
    else:
        draw_text(frame, f'Collected: {count}/{MAX_ROWS}', y, color=(120, 60, 0))
    y += 34
    if message:
        draw_text(frame, message, y, color=(0, 0, 200), thickness=2)
        y += 34
    for side, (hand, _world) in detected.items():
        orientation = get_hand_orientation(hand, side)
        draw_text(frame, f'{side}: {orientation}', y)
        y += 30
        if debug and states.get(side) and states[side][0]:
            status = states[side][0]
            draw_text(frame, 'T{Thumb} I{Index} M{Middle} R{Ring} L{Little}'.format(**status), y, color=(0, 100, 0))
            y += 30
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        draw_landmark_numbers(frame, hand)
    draw_text(frame, 'Q/Esc: stop   D: finger details', frame.shape[0] - 16)


def collect_from_video(capture, alphabet, csv_file, snapshot_dir, existing_rows, *, live, writer=None):
    """Single extraction engine for live and old video; live also records ALL frames."""
    current_count = len(existing_rows)
    seen_vectors = [
        vec for row in existing_rows
        if (vec := landmark_vector_from_row(row, alphabet)) is not None
    ]
    locked_side = saved_hand_label(existing_rows) if alphabet in ONE_HAND_ALPHABETS else None
    trackers = {'Left': FingerTracker(), 'Right': FingerTracker()}
    wrist_anchor = None  # compare against last accepted sample, not every successive frame
    show_debug = False
    start_time = time.monotonic()
    window = f'GestureFlow - {alphabet} (Q to stop)'
    total_frames = 0
    status = 'Video ended'
    written = 0
    if live:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 720)

    try:
        with mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        ) as hands:
            while current_count < MAX_ROWS:
                ok, raw_frame = capture.read()
                if not ok:
                    status = 'Camera stopped / read failed' if live else 'End of imported video'
                    break
                total_frames += 1
                frame = cv2.flip(raw_frame, 1) if live else raw_frame

                # Record the *entire* live webcam session from the first frame,
                # including the 3-second countdown, rejected frames and pauses.
                # Save clean, mirrored frames without status overlays for reuse.
                if live and writer is not None:
                    writer.write(frame)

                # Old video has no countdown, and never opens a camera/display.
                countdown = max(0.0, START_DELAY_SECONDS - (time.monotonic() - start_time)) if live else 0.0
                results, detected, states, data_row, message = process_frame(
                    frame, hands, trackers, alphabet, locked_side
                )
                accepted = False

                if countdown > 0:
                    wrist_anchor = None
                elif data_row is None:
                    wrist_anchor = None  # requires a fresh valid baseline on re-entry
                else:
                    wrists = current_wrists(detected)
                    if wrist_anchor is None:
                        wrist_anchor = wrists
                        message = 'MOVE HAND(S) IN X AND Y TO START'
                    elif not significant_wrist_movement(detected, wrist_anchor, alphabet):
                        message = 'MOVE BOTH X AND Y (EACH HAND)'
                    else:
                        candidate = [time.strftime('%Y-%m-%d %H:%M:%S')] + data_row
                        fingerprint = landmark_vector_from_row(candidate, alphabet)
                        if is_near_duplicate(fingerprint, seen_vectors):
                            message = 'NEAR-DUPLICATE SAMPLE SKIPPED'
                        else:
                            next_count = current_count + 1
                            snapshot_file = snapshot_dir / f'{alphabet}_{next_count:04d}.png'
                            if snapshot_file.exists():
                                raise FileExistsError(
                                    f'Snapshot already exists but no corresponding CSV sample: {snapshot_file}. '
                                    'Resolve the mismatch before continuing.'
                                )
                            # One stored row means exactly one stored snapshot.
                            all_hands = [detected[side][0] for side in ('Left', 'Right') if side in detected]
                            if not save_snapshot(frame.copy(), all_hands, str(snapshot_file)):
                                raise IOError(f'Unable to write snapshot {snapshot_file}')
                            try:
                                append_csv_row(csv_file, candidate)
                            except Exception:
                                snapshot_file.unlink(missing_ok=True)
                                raise
                            current_count = next_count
                            written += 1
                            seen_vectors.append(fingerprint)
                            wrist_anchor = wrists
                            if alphabet in ONE_HAND_ALPHABETS and locked_side is None:
                                locked_side = next(iter(detected))
                            accepted = True
                            message = f'SAMPLE SAVED ({current_count}/{MAX_ROWS})'
                            print(f'  {alphabet}: accepted {current_count}/{MAX_ROWS}', flush=True)

                if live:
                    display = frame.copy()
                    draw_live_status(display, alphabet, current_count, detected, states, message, show_debug, countdown)
                    cv2.imshow(window, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('d'):
                        show_debug = not show_debug
                    elif key in (ord('q'), 27):
                        status = 'Stopped by Q / Esc'
                        break
                if accepted and current_count >= MAX_ROWS:
                    status = 'Reached 100 total saved samples'
                    break
            else:
                status = 'Reached 100 total saved samples'
    finally:
        # Callers own capture/writer cleanup. Only this function owns the GUI.
        if live:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass

    return written, current_count, total_frames, status


# -------------------- MAIN PROGRAM --------------------

def main():
    print('=' * 68)
    print('GESTUREFLOW - COLLECTION DATA V4')
    print('One-hand alphabets: C, O, I, L, U, V')
    print('All other alphabets require both hands at the same time.')
    print('100 accepted samples maximum TOTAL per letter, across all sessions.')
    print('Both wrist X and Y movement must reach 0.025 since the last accepted pose.')
    print('=' * 68)

    # ASK USERNAME ONLY ONCE per process. Reuse folders on future launches.
    username = ''
    while not username:
        try:
            username = normalize_username(input('Enter user name: '))
        except ValueError as exc:
            print(exc)

    user_dir = BASE_FOLDER / username
    print(f'\nUser folder: {user_dir}')
    print('The username will NOT be requested again in this running session.')

    while True:
        alphabet = input('\nEnter alphabet A-Z (or exit): ').strip().upper()
        if alphabet == 'EXIT':
            print('GestureFlow collection program closed.')
            break
        if len(alphabet) != 1 or alphabet not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            print('Enter exactly one English alphabet A-Z, or exit.')
            continue

        user_dir, dataset_dir, snapshot_dir, csv_file = get_paths(username, alphabet)
        try:
            existing_rows = load_existing_data(csv_file)
            if alphabet in ONE_HAND_ALPHABETS:
                saved_hand_label(existing_rows)  # validation before opening camera
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            print(f'Cannot collect for {alphabet}: {exc}')
            continue

        print(f'\nAlphabet {alphabet}: {len(existing_rows)}/{MAX_ROWS} saved samples.')
        if len(existing_rows) >= MAX_ROWS:
            print('100-sample cap already reached. Choose a different alphabet.')
            continue

        mode = choose_mode(alphabet)
        if mode is None:
            print('Selection cancelled. Choose an alphabet again.')
            continue

        ensure_folders(user_dir, dataset_dir, snapshot_dir)
        if mode == 'live':
            camera = None
            writer = None
            video_path = None
            try:
                camera = open_camera()
                # Open first camera frame to size the recording, and process it too.
                ok, first_frame = camera.read()
                if not ok:
                    raise RuntimeError('Camera opened but could not read a frame.')
                writer, video_path = make_video_writer(
                    user_dir, username, alphabet, first_frame, camera
                )
                # Include the first frame in the output. No detection sample is
                # lost because the 3-second collection countdown has not elapsed.
                writer.write(cv2.flip(first_frame, 1))
                print(f'\nLive video recording: {video_path}')
                print('Camera records continuously. Q/Esc ends; D toggles finger diagnostics.')
                added, total, frames, reason = collect_from_video(
                    camera, alphabet, csv_file, snapshot_dir, existing_rows,
                    live=True, writer=writer
                )
                print(f'Finished: {reason}; new samples: {added}; total: {total}; processed frames: {frames}.')
            except (OSError, ValueError, RuntimeError, IOError) as exc:
                print(f'Live collection error: {exc}')
            finally:
                if writer is not None:
                    writer.release()
                if camera is not None:
                    camera.release()
                cv2.destroyAllWindows()

        elif mode == 'old':
            chosen_video = choose_old_video()
            if chosen_video is None:
                print('No old video selected.')
                continue
            original_capture = None
            capture = None
            try:
                # Verify decode before copying into user directory.
                original_capture = cv2.VideoCapture(str(chosen_video))
                if not original_capture.isOpened():
                    raise ValueError(f'Unable to open video: {chosen_video}')
                ok, frame = original_capture.read()
                if not ok:
                    raise ValueError('Selected video contains no readable frames.')
                original_capture.release()
                original_capture = None
                saved_video = archive_old_video(chosen_video, user_dir, username, alphabet)
                capture = cv2.VideoCapture(str(saved_video))
                if not capture.isOpened():
                    raise ValueError(f'Cannot read archived video: {saved_video}')
                print(f'\nImported video: {saved_video}')
                print('Extracting directly from video. No camera window will open.')
                added, total, frames, reason = collect_from_video(
                    capture, alphabet, csv_file, snapshot_dir, existing_rows, live=False
                )
                print(f'Finished: {reason}; new samples: {added}; total: {total}; processed frames: {frames}.')
            except (OSError, ValueError, RuntimeError, IOError) as exc:
                print(f'Old-video collection error: {exc}')
            finally:
                if original_capture is not None:
                    original_capture.release()
                if capture is not None:
                    capture.release()

        print(f'CSV:       {csv_file}')
        print(f'Snapshots: {snapshot_dir}')
        print('Choose another letter, repeat this letter if under 100, or type exit.')


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nGestureFlow closed.')
