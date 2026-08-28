import math
import cv2
import mediapipe as mp

WINDOW_NAME = "Gesture Flow - First Pipeline"
FACE_SIGN = 1

INK = (45, 45, 45)
INK_OUTLINE = (235, 235, 235)
GREEN = (0, 180, 0)
GREEN_OUTLINE = (0, 0, 0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def joint_angle(a, b, c):
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)

    dot = sum(x * y for x, y in zip(v1, v2))
    l1 = math.sqrt(sum(x * x for x in v1))
    l2 = math.sqrt(sum(x * x for x in v2))

    if not l1 or not l2:
        return 0

    cosine = max(-1, min(1, dot / (l1 * l2)))
    return math.degrees(math.acos(cosine))


def get_palm_facing(landmarks, hand_side):
    points = landmarks.landmark
    wrist, index_mcp, little_mcp = points[0], points[5], points[17]

    cross = (
        (index_mcp.x - wrist.x) * (little_mcp.y - wrist.y)
        - (index_mcp.y - wrist.y) * (little_mcp.x - wrist.x)
    )

    side_sign = 1 if hand_side == "Right" else -1
    return (
        "Front-facing"
        if cross * side_sign * FACE_SIGN > 0
        else "Back-facing"
    )


def get_hand_state(landmarks):
    points = landmarks.landmark
    wrist = points[0]

    fingers = [
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20),
    ]

    open_fingers = []
    closed_fingers = []

    for mcp, pip, tip in fingers:
        angle = joint_angle(points[mcp], points[pip], points[tip])
        pip_dist = distance(wrist, points[pip])
        tip_dist = distance(wrist, points[tip])

        open_fingers.append(
            angle >= 125 and tip_dist >= pip_dist * 1.15
        )

        closed_fingers.append(
            angle <= 150 and tip_dist <= pip_dist * 1.20
        )

    palm_width = distance(points[5], points[17])

    if not palm_width:
        return None

    thumb_extended = distance(points[4], points[5]) >= palm_width * 0.65

    if all(open_fingers):
        return "Open"

    if all(closed_fingers) and not thumb_extended:
        return "Closed"

    return None


def draw_text(frame, text, position, colour, outline, thickness=1):
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        outline, thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        colour, thickness, cv2.LINE_AA
    )


def draw_status(frame, items):
    y = 42

    draw_text(frame, "STATUS", (28, y), INK, INK_OUTLINE)
    y += 34

    for i, item in enumerate(items):
        draw_text(frame, item["side"], (28, y), INK, INK_OUTLINE)
        y += 30

        draw_text(frame, item["facing"], (28, y), INK, INK_OUTLINE)
        y += 30

        if item["state"] is not None:
            draw_text(
                frame,
                item["state"],
                (28, y),
                GREEN,
                GREEN_OUTLINE,
                2
            )
            y += 30

        if i < len(items) - 1:
            y += 14


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Camera could not be opened. Close Zoom, Teams, Camera, and browser tabs."
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        status_items = []

        if result.multi_hand_landmarks:
            for image_landmarks, world_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_hand_world_landmarks,
                result.multi_handedness,
            ):
                side = handedness.classification[0].label

                status_items.append({
                    "side": side,
                    "facing": get_palm_facing(image_landmarks, side),
                    "state": get_hand_state(world_landmarks),
                })

                mp_drawing.draw_landmarks(
                    frame,
                    image_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        draw_status(frame, status_items)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("f"):
            fullscreen = not fullscreen

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

            if not fullscreen:
                cv2.resizeWindow(WINDOW_NAME, 1280, 720)


camera.release()
cv2.destroyAllWindows()

