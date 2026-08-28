import math
import cv2
import mediapipe as mp

NAME = "Gesture Flow - First Pipeline"
INK, OUTLINE = (45,45,45), (235,235,235)
GREEN, GOUT = (0,180,0), (0,0,0)

hands = mp.solutions.hands
draw = mp.solutions.drawing_utils


def dist(a,b):
    return math.sqrt(sum((getattr(a,k)-getattr(b,k))**2 for k in "xyz"))


def angle(a,b,c):
    v1 = [getattr(a,k)-getattr(b,k) for k in "xyz"]
    v2 = [getattr(c,k)-getattr(b,k) for k in "xyz"]
    l1 = math.sqrt(sum(x*x for x in v1))
    l2 = math.sqrt(sum(x*x for x in v2))

    if not l1 or not l2:
        return 0

    return math.degrees(
        math.acos(
            max(-1, min(1, sum(x*y for x,y in zip(v1,v2))/(l1*l2)))
        )
    )


def facing(lm,side):
    p = lm.landmark
    w,i,l = p[0],p[5],p[17]

    cross = (i.x-w.x)*(l.y-w.y)-(i.y-w.y)*(l.x-w.x)

    return "Front-facing" if cross*(1 if side=="Right" else -1) > 0 else "Back-facing"


def state(lm):
    p = lm.landmark
    w = p[0]

    fingers = [(5,6,8),(9,10,12),(13,14,16),(17,18,20)]
    opened, closed = [], []

    for m,pip,t in fingers:
        a = angle(p[m],p[pip],p[t])
        pd,td = dist(w,p[pip]),dist(w,p[t])

        opened.append(a >= 125 and td >= pd*1.15)
        closed.append(a <= 150 and td <= pd*1.20)

    palm = dist(p[5],p[17])

    if not palm:
        return None

    thumb = dist(p[4],p[5]) >= palm*.65

    if all(opened):
        return "Open"

    if all(closed) and not thumb:
        return "Closed"

    return None


def text(img,s,pos,color,outline,th=1):
    cv2.putText(img,s,pos,0,.72,outline,th+3,cv2.LINE_AA)
    cv2.putText(img,s,pos,0,.72,color,th,cv2.LINE_AA)


def status(img,items):
    y = 42

    text(img,"STATUS",(28,y),INK,OUTLINE)
    y += 34

    for n,x in enumerate(items):
        text(img,x["side"],(28,y),INK,OUTLINE)
        y += 30

        text(img,x["facing"],(28,y),INK,OUTLINE)
        y += 30

        if x["state"]:
            text(img,x["state"],(28,y),GREEN,GOUT,2)
            y += 30

        if n < len(items)-1:
            y += 14


cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not cam.isOpened():
    cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise RuntimeError("Camera could not be opened.")

cam.set(3,640)
cam.set(4,480)

cv2.namedWindow(NAME,cv2.WINDOW_NORMAL)
cv2.resizeWindow(NAME,1280,720)

full = False


with hands.Hands(False,2,0,0.65,0.65) as detector:
    while True:
        ok,frame = cam.read()

        if not ok:
            break

        frame = cv2.flip(frame,1)

        result = detector.process(
            cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        )

        items = []

        if result.multi_hand_landmarks:
            for img,world,hand in zip(
                result.multi_hand_landmarks,
                result.multi_hand_world_landmarks,
                result.multi_handedness
            ):
                side = hand.classification[0].label

                items.append({
                    "side": side,
                    "facing": facing(img,side),
                    "state": state(world)
                })

                draw.draw_landmarks(
                    frame,
                    img,
                    hands.HAND_CONNECTIONS
                )

        status(frame,items)

        cv2.imshow(NAME,frame)

        key = cv2.waitKey(1) & 255

        if key == 27:
            break

        if key == ord("f"):
            full = not full

            cv2.setWindowProperty(
                NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL
            )

            if not full:
                cv2.resizeWindow(NAME,1280,720)


cam.release()
cv2.destroyAllWindows()
