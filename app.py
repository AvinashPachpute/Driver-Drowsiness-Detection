import cv2
import pygame
import time
import os
import winsound

# ---------------- Alarm Setup ----------------
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

alarm_playing = False
ALARM_FILE = "alarm.wav"

def play_alarm():
    global alarm_playing

    if not alarm_playing:
        alarm_playing = True
        print("WAKE UP! Driver is drowsy!")

        try:
            if os.path.exists(ALARM_FILE):
                pygame.mixer.music.load(ALARM_FILE)
                pygame.mixer.music.play(-1)
            else:
                winsound.Beep(2000, 800)
        except:
            winsound.Beep(2000, 800)


def stop_alarm():
    global alarm_playing

    if alarm_playing:
        alarm_playing = False
        pygame.mixer.music.stop()


# ---------------- Load Haar Cascades ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ---------------- Camera Setup ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()


# ---------------- Variables ----------------
eyes_closed_start = None
DROWSY_TIME = 2.5
status = "Awake"

print("Driver Drowsiness Detection Started")
print("Press Q to Quit")


# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    eyes_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        # Upper half (eyes region)
        upper_gray = face_gray[0:int(h/2), :]
        upper_color = face_color[0:int(h/2), :]

        eyes = eye_cascade.detectMultiScale(
            upper_gray,
            scaleFactor=1.1,
            minNeighbors=12,   # stricter detection
            minSize=(25, 25)
        )

        # 🔥 IMPORTANT FIX
        if len(eyes) >= 2:
            eyes_detected = True

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    upper_color,
                    (ex, ey),
                    (ex+ew, ey+eh),
                    (255, 0, 0),
                    2
                )

    # ---------------- Drowsiness Logic ----------------
    if eyes_detected:
        status = "Awake"
        eyes_closed_start = None
        stop_alarm()

    else:
        if len(faces) > 0:
            if eyes_closed_start is None:
                eyes_closed_start = time.time()

            closed_time = time.time() - eyes_closed_start

            if closed_time >= DROWSY_TIME:
                status = "DROWSY!"
                play_alarm()
            else:
                status = "Eyes Closing"
        else:
            status = "Face Not Detected"
            eyes_closed_start = None
            stop_alarm()

    # ---------------- Display ----------------
    if status == "DROWSY!":
        color = (0, 0, 255)
    elif status == "Awake":
        color = (0, 255, 0)
    else:
        color = (0, 255, 255)

    cv2.putText(frame, f"Status: {status}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    if status == "DROWSY!":
        cv2.putText(frame, "ALERT! WAKE UP!", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.putText(frame, "Press Q to Quit",
                (30, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ---------------- Cleanup ----------------
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()