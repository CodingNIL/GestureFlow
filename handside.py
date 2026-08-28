import cv2
import mediapipe as mp


# If front-facing and back-facing appear reversed in your test,
# change 1 to -1.
FACE_SIGN = 1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_palm_facing(hand_landmarks, hand_side):
    landmarks = hand_landmarks.landmark

    wrist = landmarks[0]
    index_mcp = landmarks[5]
    little_mcp = landmarks[17]

    cross_value = (
        (index_mcp.x - wrist.x) * (little_mcp.y - wrist.y)
        - (index_mcp.y - wrist.y) * (little_mcp.x - wrist.x)
    )

    if hand_side == "Right":
        side_sign = 1
    else:
        side_sign = -1

    if cross_value * side_sign * FACE_SIGN > 0:
        return "Front-facing"

    return "Back-facing"


def draw_status_panel(frame, status_items):
    lines = ["STATUS"]

    for item_number, item in enumerate(status_items):
        lines.append(item["side"])
        lines.append(item["facing"])

        if item_number < len(status_items) - 1:
            lines.append("")

    panel_height = 25 + len(lines) * 32

    cv2.rectangle(
        frame,
        (12, 12),
        (260, panel_height),
        (20, 20, 20),
        -1,
    )

    y_position = 42

    for line in lines:
        if line == "STATUS":
            colour = (0, 255, 255)
            thickness = 2
        else:
            colour = (255, 255, 255)
            thickness = 1

        cv2.putText(
            frame,
            line,
            (28, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            colour,
            thickness,
        )

        y_position += 32


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera could not be opened.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
) as hands:

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)

        status_items = []

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness,
            ):
                hand_side = handedness.classification[0].label

                palm_facing = get_palm_facing(
                    hand_landmarks,
                    hand_side,
                )

                status_items.append(
                    {
                        "side": hand_side,
                        "facing": palm_facing,
                    }
                )

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        draw_status_panel(frame, status_items)

        cv2.imshow("Gesture Flow - Palm Facing Status", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()