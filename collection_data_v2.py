import cv2
import math
import time
import os
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------- SETTINGS --------------------
X_VARIATION_THRESHOLD = 0.0005
Y_VARIATION_THRESHOLD = 0.0005
MAX_ROWS = 500
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

def get_finger_status(world):
    p = world.landmark

    palm_width = calculate_distance(p[5], p[17])

    if palm_width <= 0:
        return None, None

    # Finger status encoding:
    # 0 = Closed
    # 1 = Open
    #
    # Classification rule:
    # For each finger, measure the distance between its TIP landmark
    # and its BASE (MCP) joint landmark, normalized by palm width.
    #   - Closed  : tip and base landmarks nearly coincide/overlap
    #               (distance close to zero).
    #   - Open    : tip is at maximum distance from the base landmark
    #               (finger fully extended).
    #   - Anything below the OPEN threshold is treated as Closed.
    # The same rule is applied identically to the thumb and all four
    # fingers, and is applied independently per hand (Left/Right).

    # (base_landmark_index, tip_landmark_index) for each digit.
    finger_points = {
        "Thumb": (2, 4),
        "Index": (5, 8),
        "Middle": (9, 12),
        "Ring": (13, 16),
        "Little": (17, 20)
    }

    # Normalized tip-to-base distance thresholds (relative to palm width).
    CLOSED_THRESHOLD = 0.3   # Tip and base landmark nearly coincide
    OPEN_THRESHOLD = 0.9     # Tip is at (near) maximum distance from base

    status = {}
    states = []

    for name, (base_idx, tip_idx) in finger_points.items():
        base = p[base_idx]
        tip = p[tip_idx]

        normalized_distance = calculate_distance(tip, base) / palm_width

        if normalized_distance >= OPEN_THRESHOLD:
            state = 1  # Open
        else:
            state = 0  # Closed

        status[name] = state
        states.append(state)

    # Full hand position:
    # 0 = Full Closed
    # 1 = Full Open
    #
    # A hand is considered fully open only when every finger is
    # confidently classified as Open. It is fully closed only when
    # every finger is confidently classified as Closed. This is
    # evaluated separately for the Left hand and the Right hand,
    # since each is passed into this function independently.
    if all(state == 0 for state in states):
        hand_position = 0
    elif all(state == 1 for state in states):
        hand_position = 1
    else:
        hand_position = 2

    return status, hand_position

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

# -------------------- CSV COLUMNS --------------------
TWO_HAND_ALPHABETS = (
    "A", "B", "D", "E", "F", "G", "H", "J", "K",
    "M", "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"
)

ONE_HAND_ALPHABETS = ("C", "O", "I", "L", "U", "V")


def get_csv_columns(alphabet, hand_label=None):
    # For C, O, I, L, U, V, the CSV contains only the tracked hand.
    if alphabet in ONE_HAND_ALPHABETS:
        side = "L" if hand_label == "Left" else "R"

        columns = ["Timestamp"]

        for i in range(21):
            columns.extend([f"{side}x{i}", f"{side}y{i}"])

        if side == "L":
            columns.extend([
                "LT", "LI", "LM", "LR", "LL",
                "Left_Orientation",
                "Left_Hand_Position"
            ])
        else:
            columns.extend([
                "RT", "RI", "RM", "RR", "RL",
                "Right_Orientation",
                "Right_Hand_Position"
            ])

        return columns

    # Two-hand CSV order:
    # Timestamp
    # Left coordinates
    # Left finger positions
    # Left orientation
    # Full Left hand position
    # Right coordinates
    # Right finger positions
    # Right orientation
    # Full Right hand position

    columns = ["Timestamp"]

    for i in range(21):
        columns.extend([f"Lx{i}", f"Ly{i}"])

    columns.extend([
        "LT", "LI", "LM", "LR", "LL",
        "Left_Orientation",
        "Left_Hand_Position"
    ])

    for i in range(21):
        columns.extend([f"Rx{i}", f"Ry{i}"])

    columns.extend([
        "RT", "RI", "RM", "RR", "RL",
        "Right_Orientation",
        "Right_Hand_Position"
    ])

    return columns

def create_empty_row(alphabet, hand_label=None):
    return [pd.NA] * (len(get_csv_columns(alphabet, hand_label)) - 1)


def add_hand_to_row(row, label, hand, finger_status, hand_position, alphabet):
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]

    if alphabet in ONE_HAND_ALPHABETS:
        # Only the tracked hand is written to the row.
        for i, point in enumerate(hand.landmark):
            row[i * 2] = point.x
            row[i * 2 + 1] = point.y

        for i, name in enumerate(finger_names):
            row[42 + i] = finger_status[name]

        orientation = get_hand_orientation(hand, label)
        orientation_code = {"Front": "F", "Back": "B", "Side": "S"}[orientation]
        row[47] = orientation_code
        row[48] = hand_position
        return

    # Two-hand alphabet.
    # Left coordinates: 0-41
    # Left fingers: 42-46
    # Left orientation: 47
    # Left full hand position: 48
    #
    # Right coordinates: 49-90
    # Right fingers: 91-95
    # Right orientation: 96
    # Right full hand position: 97

    start = 0 if label == "Left" else 49

    for i, point in enumerate(hand.landmark):
        row[start + i * 2] = point.x
        row[start + i * 2 + 1] = point.y

    if label == "Left":
        finger_start = 42
        orientation_index = 47
        position_index = 48
    else:
        finger_start = 91
        orientation_index = 96
        position_index = 97

    for i, name in enumerate(finger_names):
        row[finger_start + i] = finger_status[name]

    orientation = get_hand_orientation(hand, label)
    orientation_code = {"Front": "F", "Back": "B", "Side": "S"}[orientation]
    row[orientation_index] = orientation_code
    row[position_index] = hand_position

# -------------------- CSV FUNCTIONS --------------------
def load_existing_data(csv_file):

    if os.path.exists(csv_file):

        try:
            dataframe = pd.read_csv(csv_file)
            return dataframe.values.tolist()
        except Exception:
            print("Warning: Could not read existing CSV.")
            return []

    return []

def save_csv(data_rows, csv_file, alphabet, hand_label=None):
    csv_columns = get_csv_columns(alphabet, hand_label)
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

    # Take the alphabet snapshot only once for each alphabet.
    # If it already exists, no new snapshot will be taken.
    snapshot_file = os.path.join(
        snapshot_folder,
        f"{alphabet}.png"
    )
    snapshot_taken = os.path.exists(snapshot_file)
    image_captured_flash = False
    image_captured_flash_until = 0

    existing_hand_label = None

    if alphabet in ONE_HAND_ALPHABETS and os.path.exists(csv_file):
        try:
            existing_columns = pd.read_csv(csv_file, nrows=0).columns.tolist()

            if "Lx0" in existing_columns:
                existing_hand_label = "Left"
            elif "Rx0" in existing_columns:
                existing_hand_label = "Right"
        except Exception:
            existing_hand_label = None

    if existing_data:
        print(f"\nExisting {alphabet}.csv found.")
        print(f"Existing rows: {len(existing_data)}")
        print("New data will be appended.")
    else:
        print(f"\nCreating new {alphabet}.csv")

    data_rows = []                                                           # SESSION VARIABLES
    total_rows = 0
    last_wrist = {"Left": None,"Right": None}
    one_hand_collected_label = existing_hand_label if alphabet in ONE_HAND_ALPHABETS else None
    hand_switch_warning = None

    # If the alphabet already has 200+ saved samples but its snapshot was
    # never taken, the next valid collected sample will create the snapshot.
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
        model_complexity=0,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    ) as hands:

        while True:
            success, frame = camera.read()

            if not success:
                break
            
            frame = cv2.flip(frame,1)                                             # Mirror camera

            # Separate clean frame for the alphabet snapshot.
            # This frame contains no status text.
            snapshot_frame = frame.copy()

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

                    # Finger positions and overall hand position are
                    # calculated for CSV storage only. They are not displayed
                    # on the camera screen.
                    finger_status, hand_position = get_finger_status(world)

                    # Draw landmarks on the normal camera frame.
                    mp_drawing.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )
                    draw_landmark_numbers(frame, hand)

                    # Draw the same landmarks on the clean snapshot frame.
                    # No status text is added to snapshot_frame.
                    mp_drawing.draw_landmarks(
                        snapshot_frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )
                    draw_landmark_numbers(snapshot_frame, hand)

                # ---------------------------------------------------------
                # MANDATORY HAND CONDITION
                #
                # C, O, I, L, U, V:
                #   Exactly ONE hand is accepted.
                #   The CSV for that alphabet contains ONLY the tracked
                #   hand's columns. No NA sample is created.
                #
                # All other alphabets:
                #   BOTH hands must be present.
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
                            # present, so only that hand becomes the sample.
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
                                elif label != one_hand_collected_label:
                                    # Wrong hand: do not collect data.
                                    # Store the warning so it is drawn on the
                                    # camera every frame while the wrong hand
                                    # remains selected.
                                    hand_switch_warning = (
                                        f"{one_hand_collected_label} HAND REQUIRED"
                                    )
                                    last_wrist = {"Left": None, "Right": None}
                                    continue

                                finger_status, hand_position = get_finger_status(world)

                                if finger_status:
                                    current_row = create_empty_row(alphabet, label)

                                    add_hand_to_row(
                                        current_row,
                                        label,
                                        hand,
                                        finger_status,
                                        hand_position,
                                        alphabet
                                    )

                                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                    data_rows.append(
                                        [timestamp] + current_row
                                    )
                                    total_rows += 1

                                    # Save a clean snapshot at 200th sample.
                                    # It is taken only once for this alphabet.
                                    # The snapshot contains only the camera
                                    # image and hand landmarks.
                                    if existing_sample_count + total_rows >= 200 and not snapshot_taken:
                                        if cv2.imwrite(snapshot_file, snapshot_frame):
                                            snapshot_taken = True
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
                                    finger_status, hand_position = get_finger_status(world)

                                    if finger_status:
                                        add_hand_to_row(
                                            current_row,
                                            label,
                                            hand,
                                            finger_status,
                                            hand_position,
                                            alphabet
                                        )

                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                data_rows.append(
                                    [timestamp] + current_row
                                )
                                total_rows += 1

                                # Save a clean snapshot at 200th sample.
                                # It is taken only once for this alphabet.
                                # The snapshot contains only the camera
                                # image and hand landmarks.
                                if existing_sample_count + total_rows >= 200 and not snapshot_taken:
                                    if cv2.imwrite(snapshot_file, snapshot_frame):
                                        snapshot_taken = True
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

            if key in (ord("q"),27):
                print(f"\nCollection stopped for {alphabet}.")
                break
    
    if data_rows:                                                 # SAVE CSV
        combined_data = (existing_data + data_rows)
        try:
            hand_label = None

            if alphabet in ONE_HAND_ALPHABETS:
                # Reuse the existing CSV's hand type when appending.
                if existing_hand_label:
                    hand_label = existing_hand_label
                elif data_rows:
                    # One-hand rows have 49 values after the timestamp.
                    # The actual hand is determined from the first row's
                    # original collection schema.
                    hand_label = one_hand_collected_label

            save_csv(combined_data, csv_file, alphabet, hand_label)
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
    