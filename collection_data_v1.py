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
MAX_ROWS = 300
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

    cross_z = (index.x - wrist.x) * (pinky.y - wrist.y) - (index.y - wrist.y) * (pinky.x - wrist.x)
    sign = 1 if label == "Right" else -1

    return "Front" if cross_z * sign > 0 else "Back"

def get_finger_status(world):
    p = world.landmark
    states = []
    palm_width = calculate_distance(p[5], p[17])

    if not palm_width:
        return None

    thumb_dist = calculate_distance(p[4], p[5])      # Thumb

    if thumb_dist >= palm_width * 0.9:
        states.append(1)
    elif thumb_dist <= palm_width * 0.65:
        states.append(0)
    else:
        states.append(-1)

    fingers = [(5, 6, 8),(9, 10, 12),(13, 14, 16),(17, 18, 20)]     # Index, middle, ring and little fingers

    for mcp, pip, tip in fingers:
        angle = calculate_angle(p[mcp],p[pip],p[tip])
        pip_dist = calculate_distance(p[0],p[pip])
        tip_dist = calculate_distance(p[0],p[tip])

        if angle >= 125 and tip_dist >= pip_dist * 1.15:
            states.append(1)
        elif angle <= 150 and tip_dist <= pip_dist * 1.20:
            states.append(0)
        else:
            states.append(-1)

    if all(s == 1 for s in states):
        return "Open"
    if all(s == 0 for s in states):
        return "Closed"
    if any(s in (0, 1) for s in states):
        open_count = sum(s == 1 for s in states)
        return f"Partial ({open_count}/5 open)"

    return None

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
columns = ["Timestamp"]                                                 # Timestamp + 21 landmarks × X/Y for Left and Right hands

for side in ("L", "R"):

    for i in range(21):
        columns.extend([f"{side}x{i}",f"{side}y{i}"])

columns.extend(["Left_Orientation","Right_Orientation"])                 # Orientation columns

def create_empty_row():
    return [pd.NA] * 84  # 42 coordinates for each hand = 84 coordinates

def add_hand_to_row(row, label, hand):                                    # Left hand starts at index 0 & Right hand starts at index 42
    start = 0 if label == "Left" else 42        

    for i, point in enumerate(hand.landmark):
        row[start + i * 2] = point.x
        row[start + i * 2 + 1] = point.y

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

def save_csv(data_rows, csv_file):
    dataframe = pd.DataFrame(data_rows,columns=columns)
    dataframe.to_csv(csv_file,index=False,na_rep="NA")

username = input("Enter user name: ").strip()                               # USER NAME

if not username:
    print("User name cannot be empty.")
    raise SystemExit

user_folder = f"{username}_Dataset"                                         # CREATE USER FOLDER
os.makedirs(user_folder,exist_ok=True)
print(f"\nData folder : {user_folder}")

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

    if existing_data:
        print(f"\nExisting {alphabet}.csv found.")
        print(f"Existing rows: {len(existing_data)}")
        print("New data will be appended.")
    else:
        print(f"\nCreating new {alphabet}.csv")

    data_rows = []                                                           # SESSION VARIABLES
    total_rows = 0
    last_wrist = {"Left": None,"Right": None}

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
            elapsed = (time.time() - start_time)
            capturing = (elapsed >= START_DELAY_SECONDS) 

            rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)                     # BGR → RGB
            results = hands.process(rgb_frame)

            y = 42                                                                # DISPLAY STATUS
            draw_text(frame,f"Collect Data : {alphabet}",y)
            y += 34

            if capturing:
                draw_text(frame,f"Collected: {total_rows}/{MAX_ROWS}",y,color=(120, 60, 0))
            else:
                seconds_left = (START_DELAY_SECONDS - elapsed)
                draw_text(frame,f"Starting in {seconds_left:.1f}s...",y,color=(0, 0, 200))

            y += 34

            if results.multi_hand_landmarks:                                      # HAND DETECTION
                current_row = create_empty_row()                                  # One row can contain left, right or both hands
                triggered = False
                left_orientation = "NA"
                right_orientation = "NA"

                for (hand,world,handedness) in zip(results.multi_hand_landmarks,results.multi_hand_world_landmarks,results.multi_handedness):
                    label = (handedness.classification[0].label)                   # LEFT / RIGHT
                    draw_text(frame,label,y)
                    y += 30

                    orientation = get_hand_orientation(hand,label)                  # ORIENTATION
                    draw_text(frame,orientation,y)
                    y += 30

                    if label == "Left":
                        left_orientation = orientation
                    else:
                        right_orientation = orientation

                    status = get_finger_status( world)                                # FINGER STATUS
                                       
                    if status:
                        draw_text(frame,status,y,color=(0, 0, 0),outline=(0, 180, 0),thickness=2)
                        y += 30

                    mp_drawing.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)    # DRAW LANDMARKS
                    draw_landmark_numbers(frame,hand)

                    wrist = hand.landmark[0]                                           # WRIST MOVEMENT   # Landmark 0 = wrist
                    current_wrist = (wrist.x,wrist.y)

                    if (capturing and last_wrist[label] is not None):
                        previous_wrist = (last_wrist[label])
                        x_change = abs(current_wrist[0] - previous_wrist[0])
                        y_change = abs(current_wrist[1] - previous_wrist[1])

                        if (x_change >= X_VARIATION_THRESHOLD or y_change >= Y_VARIATION_THRESHOLD):       # Capture this hand if wrist moved
                            add_hand_to_row(current_row,label,hand)
                            triggered = True
                            draw_text(frame,f"{label} CAPTURED",y,color=(0, 0, 0),outline=(0, 140, 255),thickness=2)
                            y += 30

                    last_wrist[label] = current_wrist          # IMPORTANT:Previous wrist is updated every frame
                    y += 44

                if capturing and triggered:                    # SAVE CAPTURE EVENT
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    data_rows.append([timestamp]+ current_row + [left_orientation,right_orientation])
                    total_rows += 1

            else:
                last_wrist = {"Left": None,"Right": None}       # No hand detected : Reset previous wrist positions

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
            save_csv(combined_data,csv_file)
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