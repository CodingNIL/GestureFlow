import math

import cv2
import mediapipe as mp


WINDOW_NAME = "Gesture Flow - First Pipeline"

# If Front-facing and Back-facing are reversed, change 1 to -1.
FACE_SIGN = 1

# OpenCV uses BGR colour order, not RGB.
INK = (45, 45, 45)          # Dark ink / charcoal
INK_OUTLINE = (235, 235, 235)
GREEN = (0, 180, 0)         # Green for Open and Closed
GREEN_OUTLINE = (0, 0, 0)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def distance(point_a, point_b):
    """Returns 3D distance between two MediaPipe landmarks."""
    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
        + (point_a.z - point_b.z) ** 2
    )


def joint_angle(point_a, point_b, point_c):
    """Returns the 3D bend angle at point_b."""
    vector_1 = (
        point_a.x - point_b.x,
        point_a.y - point_b.y,
        point_a.z - point_b.z,
    )

    vector_2 = (
        point_c.x - point_b.x,
        point_c.y - point_b.y,
        point_c.z - point_b.z,
    )

    dot_product = (
        vector_1[0] * vector_2[0]
        + vector_1[1] * vector_2[1]
        + vector_1[2] * vector_2[2]
    )

    length_1 = math.sqrt(
        vector_1[0] ** 2
        + vector_1[1] ** 2
        + vector_1[2] ** 2
    )

    length_2 = math.sqrt(
        vector_2[0] ** 2
        + vector_2[1] ** 2
        + vector_2[2] ** 2
    )

    if length_1 == 0 or length_2 == 0:
        return 0

    cosine = dot_product / (length_1 * length_2)
    cosine = max(-1, min(1, cosine))

    return math.degrees(math.acos(cosine))


def get_palm_facing(image_landmarks, hand_side):
    """Returns Front-facing or Back-facing."""
    landmarks = image_landmarks.landmark

    wrist = landmarks[0]
    index_mcp = landmarks[5]
    little_mcp = landmarks[17]

    cross_value = (
        (index_mcp.x - wrist.x) * (little_mcp.y - wrist.y)
        - (index_mcp.y - wrist.y) * (little_mcp.x - wrist.x)
    )

    side_sign = 1 if hand_side == "Right" else -1

    if cross_value * side_sign * FACE_SIGN > 0:
        return "Front-facing"

    return "Back-facing"


def get_hand_state(world_landmarks):
    """
    Returns:
        Open   -> all four main fingers are extended
        Closed -> all four main fingers are folded and thumb is not extended
        None   -> mixed pose, so Open/Closed is not shown
    """
    landmarks = world_landmarks.landmark
    wrist = landmarks[0]

    finger_rules = [
        (5, 6, 8),      # Index: MCP, PIP, TIP
        (9, 10, 12),    # Middle
        (13, 14, 16),   # Ring
        (17, 18, 20),   # Little
    ]

    long_fingers_open = []
    long_fingers_closed = []

    for mcp_index, pip_index, tip_index in finger_rules:
        angle = joint_angle(
            landmarks[mcp_index],
            landmarks[pip_index],
            landmarks[tip_index],
        )

        pip_distance = distance(wrist, landmarks[pip_index])
        tip_distance = distance(wrist, landmarks[tip_index])

        # An extended finger is fairly straight and its tip is farther
        # from the wrist than its middle joint.
        is_open = (
            angle >= 125
            and tip_distance >= pip_distance * 1.15
        )

        # A folded finger is bent and its tip is near its middle joint.
        is_closed = (
            angle <= 150
            and tip_distance <= pip_distance * 1.20
        )

        long_fingers_open.append(is_open)
        long_fingers_closed.append(is_closed)

    palm_width = distance(landmarks[5], landmarks[17])

    if palm_width == 0:
        return None

    # Used only to prevent Thumbs-up from being labelled Closed.
    thumb_distance = distance(landmarks[4], landmarks[5])

    thumb_is_extended = (
        thumb_distance >= palm_width * 0.65
    )

    # All four main fingers must be extended.
    if all(long_fingers_open):
        return "Open"

    # All four main fingers must be folded and thumb must not be raised.
    if all(long_fingers_closed) and not thumb_is_extended:
        return "Closed"

    return None


def draw_text(frame, text, position, colour, outline_colour, thickness=1):
    """
    Draws text directly on the camera frame.
    No background rectangle is used.
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        outline_colour,
        thickness + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def draw_status(frame, status_items):
    """
    Shows only:

    STATUS
    Left or Right
    Front-facing or Back-facing
    Open or Closed, only for a complete hand state
    """
    y_position = 42

    draw_text(
        frame,
        "STATUS",
        (28, y_position),
        INK,
        INK_OUTLINE,
        1,
    )

    y_position += 34

    for item_number, item in enumerate(status_items):
        draw_text(
            frame,
            item["side"],
            (28, y_position),
            INK,
            INK_OUTLINE,
        )

        y_position += 30

        draw_text(
            frame,
            item["facing"],
            (28, y_position),
            INK,
            INK_OUTLINE,
        )

        y_position += 30

        # For a mixed hand pose, state is None, so nothing is shown.
        if item["state"] is not None:
            draw_text(
                frame,
                item["state"],
                (28, y_position),
                GREEN,
                GREEN_OUTLINE,
                2,
            )

            y_position += 30

        if item_number < len(status_items) - 1:
            y_position += 14


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Camera could not be opened. Close Zoom, Teams, Camera, and browser tabs."
    )

# A suitable resolution for your i3 and 8 GB RAM laptop.
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Creates a larger, resizable camera window.
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

fullscreen = False


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

        # Mirror effect for normal webcam behaviour.
        frame = cv2.flip(frame, 1)

        # OpenCV uses BGR, while MediaPipe requires RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)

        status_items = []

        if result.multi_hand_landmarks:
            for image_landmarks, world_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_hand_world_landmarks,
                result.multi_handedness,
            ):
                hand_side = handedness.classification[0].label

                status_items.append(
                    {
                        "side": hand_side,
                        "facing": get_palm_facing(
                            image_landmarks,
                            hand_side,
                        ),
                        "state": get_hand_state(world_landmarks),
                    }
                )

                mp_drawing.draw_landmarks(
                    frame,
                    image_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        draw_status(frame, status_items)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        # Q or Esc closes Gesture Flow.
        if key in (ord("q"), 27):
            break

        # F switches fullscreen on and off.
        if key == ord("f"):
            fullscreen = not fullscreen

            if fullscreen:
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
            else:
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL,
                )

                cv2.resizeWindow(WINDOW_NAME, 1280, 720)


camera.release()
cv2.destroyAllWindows()