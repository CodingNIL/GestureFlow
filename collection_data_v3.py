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
X_VARIATION_THRESHOLD = 0.0005
Y_VARIATION_THRESHOLD = 0.0005
MAX_ROWS = 100
SNAPSHOT_EVERY = 1          # save a picture every 50 samples (500 samples -> 10 pictures)
START_DELAY_SECONDS = 3

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

# -------------------- CSV FUNCTIONS --------------------
def load_existing_data(csv_file):

    if os.path.exists(csv_file):

        try:
            dataframe = pd.read_csv(csv_file)
        except Exception:
            print("Warning: Could not read existing CSV.")
            return []

        # Files saved with the old column layout cannot be mixed with the
        # new layout, so the old file is kept as a backup and a new one
        # is started.
        if list(dataframe.columns) != get_csv_columns():
            backup_file = csv_file.replace(
                ".csv",
                f"_old_format_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            )
            try:
                os.replace(csv_file, backup_file)
                print(f"Warning: {csv_file} uses the old column format.")
                print(f"It was renamed to {backup_file}. Starting a new file.")
            except PermissionError:
                print(f"Warning: {csv_file} uses the old column format and "
                      "could not be renamed. Close it and run again.")
                raise SystemExit
            return []

        return dataframe.values.tolist()

    return []

def save_csv(data_rows, csv_file):
    csv_columns = get_csv_columns()
    dataframe = pd.DataFrame(data_rows, columns=csv_columns)
    dataframe.to_csv(csv_file, index=False, na_rep="NA")

username = input("Enter user name: ").strip()                               # USER NAME

if not username:
    print("User name cannot be empty.")
    raise SystemExit

user_folder = f"{username}_Dataset"                                         # CREATE USER FOLDER
os.makedirs(user_folder,exist_ok=True)

snapshot_folder = f"{username}_Snapshots"
os.makedirs(snapshot_folder, exist_ok=True)

print(f"\nData folder : {user_folder}")

print("\nCSV COLUMN INFORMATION")
print("LT = Left Thumb")
print("LI = Left Index")
print("LM = Left Middle")
print("LR = Left Ring")
print("LL = Left Little")
print("RT = Right Thumb")
print("RI = Right Index")
print("RM = Right Middle")
print("RR = Right Ring")
print("RL = Right Little")
print("0 = Closed")
print("1 = Open")
print("Hand Orientation:")
print("F = Front")
print("B = Back")
print("S = Side")
print("Full Hand Position: 0 = Full Closed, 1 = Full Open")
print("Wrist_Dist_X / Wrist_Dist_Y = horizontal / vertical distance between the two wrists")
print("Thumb_Index_Dist = distance between landmarks 4 and 8 (per hand)")
print("Dist_4_5 / Dist_4_6 = distance between landmarks 4-5 and 4-6 (per hand)")
print("Left_Right_Index_Dist = distance between left and right index fingertips (landmark 8)")
print("One-hand alphabets: the unused hand's columns and cross-hand distances are 0")
print("Camera keys: d = show/hide live finger states, q or Esc = stop")
print(f"A snapshot is saved every {SNAPSHOT_EVERY} samples in {username}_Snapshots")


while True:                                                                 # ALPHABET COLLECTION LOOP

    alphabet = input("\nEnter ISL alphabet (A-Z) " "or type 'exit' to quit: ").strip().upper()

    if alphabet == "EXIT":                                                  # Exit whole program
        print("Collection program closed.")
        break
    if len(alphabet) != 1 or not alphabet.isalpha():                        # Check alphabet
        print("Please enter a single alphabet from A to Z.")
        continue

    csv_file = os.path.join(user_folder,f"{alphabet}.csv")                  # CSV FILE PATH

    existing_data = load_existing_data(csv_file)                            # LOAD EXISTING DATA

    # A snapshot is saved every SNAPSHOT_EVERY samples, named
    # <alphabet>_<sample number>.png (for example C_0050.png).
    image_captured_flash = False
    image_captured_flash_until = 0

    if existing_data:
        print(f"\nExisting {alphabet}.csv found.")
        print(f"Existing rows: {len(existing_data)}")
        print("New data will be appended.")
    else:
        print(f"\nCreating new {alphabet}.csv")

    data_rows = []                                                           # SESSION VARIABLES
    total_rows = 0
    last_wrist = {"Left": None,"Right": None}
    one_hand_collected_label = None
    hand_switch_warning = None
    finger_trackers = {"Left": FingerTracker(), "Right": FingerTracker()}
    show_finger_debug = False                                                # press "d" to toggle

    # Sample numbers continue from the rows already in the CSV, so snapshots
    # from a later session never overwrite earlier ones.
    existing_sample_count = len(existing_data)

    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)                               # CAMERA

    if not camera.isOpened():
        camera = cv2.VideoCapture(0) 
    if not camera.isOpened():
        print("Camera could not be opened.")
        continue

    camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    window_name = (f"Collect Data : {alphabet}")
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name,1280,720)
    start_time = time.time()

    with mp_hands.Hands(                                                       # MEDIAPIPE
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,                  # full model: more accurate landmarks than 0
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = camera.read()

            if not success:
                break
            
            frame = cv2.flip(frame,1)                                             # Mirror camera

            # Clean copy of the camera image (no text, no landmarks) used for
            # the periodic snapshot.
            clean_frame = frame.copy()

            elapsed = (time.time() - start_time)
            capturing = (elapsed >= START_DELAY_SECONDS) 

            rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)                     # BGR → RGB
            results = hands.process(rgb_frame)

            y = 42                                                                # DISPLAY STATUS
            draw_text(frame,f"Collect Data : {alphabet}",y)
            y += 34

            if capturing:
                # Keep the collection progress in the status area.
                draw_text(
                    frame,
                    f"Collected: {total_rows}/{MAX_ROWS}",
                    y,
                    color=(120, 60, 0)
                )
            else:
                # Keep the countdown before data collection starts.
                seconds_left = (START_DELAY_SECONDS - elapsed)
                draw_text(
                    frame,
                    f"Starting in {seconds_left:.1f}s...",
                    y,
                    color=(0, 0, 200)
                )

            y += 34

            if results.multi_hand_landmarks:                                      # HAND DETECTION
                detected_hands = {}
                hand_status = {}

                # Collect all currently detected hands first.
                for (hand, world, handedness) in zip(
                    results.multi_hand_landmarks,
                    results.multi_hand_world_landmarks,
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label
                    detected_hands[label] = (hand, world)

                    draw_text(frame, label, y)
                    y += 30

                    orientation = get_hand_orientation(hand, label)
                    draw_text(frame, orientation, y)
                    y += 30

                    # Finger states are computed ONCE per hand per frame (3D rule
                    # + majority vote over the last few frames) and reused below
                    # when the CSV row is built.
                    finger_status, hand_position = finger_trackers[label].update(world)
                    hand_status[label] = (finger_status, hand_position)

                    # Optional live check (press "d"): the finger states that
                    # will be written to the CSV. T I M R L = Thumb, Index,
                    # Middle, Ring, Little. 1 = open, 0 = closed.
                    if show_finger_debug and finger_status:
                        draw_text(
                            frame,
                            "T{Thumb} I{Index} M{Middle} R{Ring} L{Little}".format(**finger_status),
                            y,
                            color=(0, 100, 0)
                        )
                        y += 30

                    # Draw landmarks on the normal camera frame.
                    mp_drawing.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )
                    draw_landmark_numbers(frame, hand)

                # A hand that left the picture starts with a fresh vote window.
                for tracked_label in finger_trackers:
                    if tracked_label not in detected_hands:
                        finger_trackers[tracked_label].reset()

                # ---------------------------------------------------------
                # MANDATORY HAND CONDITION
                #
                # C, O, I, L, U, V:
                #   Exactly ONE hand is accepted.
                #   The tracked hand's columns are filled and ALL columns
                #   of the other hand are 0. Wrist distance is 0.
                #
                # All other alphabets:
                #   BOTH hands must be present. Wrist-to-wrist distance
                #   (X and Y) is stored with every sample.
                # ---------------------------------------------------------
                one_hand_alphabet = alphabet in ONE_HAND_ALPHABETS
                both_hands_present = (
                    "Left" in detected_hands and "Right" in detected_hands
                )
                exactly_one_hand = len(detected_hands) == 1

                if one_hand_alphabet:
                    valid_hand_condition = exactly_one_hand
                else:
                    valid_hand_condition = both_hands_present

                if capturing:
                    if not valid_hand_condition:
                        if one_hand_alphabet:
                            draw_text(
                                frame,
                                "EXACTLY ONE HAND REQUIRED",
                                y,
                                color=(0, 0, 200),
                                thickness=2
                            )
                        else:
                            draw_text(
                                frame,
                                "BOTH HANDS REQUIRED",
                                y,
                                color=(0, 0, 200),
                                thickness=2
                            )
                        y += 32

                        last_wrist = {"Left": None, "Right": None}

                    else:
                        if one_hand_alphabet and hand_switch_warning is not None:
                            draw_text(
                                frame,
                                hand_switch_warning,
                                y,
                                color=(0, 0, 200),
                                outline=(255, 255, 255),
                                thickness=2
                            )
                            y += 32

                        # Check wrist movement for the accepted hand(s).
                        moved_labels = []

                        for label, (hand, world) in detected_hands.items():
                            wrist = hand.landmark[0]
                            current_wrist = (wrist.x, wrist.y)

                            if last_wrist[label] is not None:
                                previous_wrist = last_wrist[label]
                                x_change = abs(current_wrist[0] - previous_wrist[0])
                                y_change = abs(current_wrist[1] - previous_wrist[1])

                                if (
                                    x_change >= X_VARIATION_THRESHOLD
                                    or y_change >= Y_VARIATION_THRESHOLD
                                ):
                                    moved_labels.append(label)

                            last_wrist[label] = current_wrist

                        if one_hand_alphabet and hand_switch_warning is not None:
                            if (
                                len(detected_hands) == 1
                                and one_hand_collected_label in detected_hands
                            ):
                                hand_switch_warning = None

                        if moved_labels:
                            # For one-hand alphabets, exactly one hand is
                            # present, so only that hand becomes the sample
                            # (the other hand's columns stay 0).
                            # For two-hand alphabets, both hands are present,
                            # so one complete two-hand sample is created.
                            if one_hand_alphabet:
                                label = moved_labels[0]
                                hand, world = detected_hands[label]

                                # For a one-hand gesture, the first hand used
                                # for collection is locked for this session.
                                # If the user changes hands, data collection
                                # stops for that frame and a warning is shown.
                                if one_hand_collected_label is None:
                                    one_hand_collected_label = label

                                if label != one_hand_collected_label:
                                    # Wrong hand: do not collect data.
                                    # Store the warning so it is drawn on the
                                    # camera every frame while the wrong hand
                                    # remains selected.
                                    hand_switch_warning = (
                                        f"{one_hand_collected_label} HAND REQUIRED"
                                    )
                                    last_wrist = {"Left": None, "Right": None}

                                else:
                                    finger_status, hand_position = hand_status[label]

                                    if finger_status:
                                        # Row starts as all 0, so the other
                                        # hand's columns remain 0.
                                        current_row = create_empty_row(alphabet)

                                        add_hand_to_row(
                                            current_row,
                                            label,
                                            hand,
                                            finger_status,
                                            hand_position
                                        )
                                        add_wrist_distance(current_row, detected_hands)

                                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                        data_rows.append(
                                            [timestamp] + current_row
                                        )
                                        total_rows += 1

                                        # Snapshot every SNAPSHOT_EVERY samples.
                                        # It contains only the camera image and
                                        # hand landmarks (no status text).
                                        sample_number = existing_sample_count + total_rows
                                        if sample_number % SNAPSHOT_EVERY == 0:
                                            snapshot_file = os.path.join(
                                                snapshot_folder,
                                                f"{alphabet}_{sample_number:04d}.png"
                                            )
                                            if save_snapshot(clean_frame, results.multi_hand_landmarks, snapshot_file):
                                                image_captured_flash = True
                                                image_captured_flash_until = time.time() + 2.0
                                                print(f"Snapshot saved: {snapshot_file}")

                                        draw_text(
                                            frame,
                                            f"{label} CAPTURED",
                                            y,
                                            color=(0, 0, 0),
                                            outline=(0, 140, 255),
                                            thickness=2
                                        )
                                        y += 30

                            else:
                                # Two-hand alphabets always save both hands
                                # together in one complete row.
                                current_row = create_empty_row(alphabet)

                                for label, (hand, world) in detected_hands.items():
                                    finger_status, hand_position = hand_status[label]

                                    if finger_status:
                                        add_hand_to_row(
                                            current_row,
                                            label,
                                            hand,
                                            finger_status,
                                            hand_position
                                        )

                                # Wrist-to-wrist distance (X and Y).
                                add_wrist_distance(current_row, detected_hands)

                                # Distance between left and right index fingertips.
                                add_left_right_index_distance(
                                    current_row, detected_hands
                                )

                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                data_rows.append(
                                    [timestamp] + current_row
                                )
                                total_rows += 1

                                # Snapshot every SNAPSHOT_EVERY samples.
                                # It contains only the camera image and
                                # hand landmarks (no status text).
                                sample_number = existing_sample_count + total_rows
                                if sample_number % SNAPSHOT_EVERY == 0:
                                    snapshot_file = os.path.join(
                                        snapshot_folder,
                                        f"{alphabet}_{sample_number:04d}.png"
                                    )
                                    if save_snapshot(clean_frame, results.multi_hand_landmarks, snapshot_file):
                                        image_captured_flash = True
                                        image_captured_flash_until = time.time() + 2.0
                                        print(f"Snapshot saved: {snapshot_file}")

                                for label in moved_labels:
                                    draw_text(
                                        frame,
                                        f"{label} CAPTURED",
                                        y,
                                        color=(0, 0, 0),
                                        outline=(0, 140, 255),
                                        thickness=2
                                    )
                                    y += 30

                else:
                    # During the 3-second starting delay, keep wrist tracking reset.
                    last_wrist = {"Left": None, "Right": None}

            else:
                last_wrist = {"Left": None,"Right": None}       # No hand detected : Reset previous wrist positions
                for tracker in finger_trackers.values():
                    tracker.reset()
                if alphabet in ONE_HAND_ALPHABETS:
                    hand_switch_warning = None

            # Flash "IMAGE CAPTURED" in the fixed camera status area.
            # It appears only when a new PNG was actually saved.
            if image_captured_flash:
                if time.time() <= image_captured_flash_until:
                    draw_text(
                        frame,
                        "IMAGE CAPTURED",
                        110,
                        color=(0, 120, 0),
                        outline=(255, 255, 255),
                        thickness=2
                    )
                else:
                    image_captured_flash = False

            cv2.imshow(window_name,frame)                       # SHOW CAMERA
            
            if (capturing and total_rows >= MAX_ROWS):          # MAXIMUM ROWS
                print(f"\nFinished collecting " f"{MAX_ROWS} new rows for {alphabet}.")
                break
            
            key = cv2.waitKey(1) & 0xFF                          # Q / ESC

            if key == ord("d"):                                  # show / hide finger states
                show_finger_debug = not show_finger_debug

            if key in (ord("q"),27):
                print(f"\nCollection stopped for {alphabet}.")
                break
    
    if data_rows:                                                 # SAVE CSV
        combined_data = (existing_data + data_rows)
        try:
            save_csv(combined_data, csv_file)
            print(f"\nSaved: {csv_file}")
            print(f"New rows added: {len(data_rows)}")
            print(f"Total rows in file: " f"{len(combined_data)}")
        except PermissionError:
            print(f"\nCould not save {alphabet}.csv.")
            print("Please close the CSV and run again.")

    else:
        print(f"\nNo new data collected for {alphabet}.")
    
    camera.release()                                                # CLEANUP
    cv2.destroyAllWindows()
    print(f"\n{alphabet} collection completed.")