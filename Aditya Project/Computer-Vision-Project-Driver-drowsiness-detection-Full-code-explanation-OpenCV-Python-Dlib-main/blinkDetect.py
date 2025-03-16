import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import queue
import mediapipe as mp

# Constants
FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460
thresh = 0.27
sound_path = "alarm.wav"
blinkTime = 0.15  # 150ms
drowsyTime = 1.5  # 1200ms
ALARM_ON = False

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark indices for eyes
leftEyeIndex = [362, 385, 387, 263, 373, 380]
rightEyeIndex = [33, 160, 158, 133, 153, 144]

# Thread queue
threadStatusQ = queue.Queue()

# Gamma correction
GAMMA = 1.5
invGamma = 1.0 / GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")


def gamma_correction(image):
    return cv2.LUT(image, table)


def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def soundAlert(path, threadStatusQ):
    while True:
        if not threadStatusQ.empty():
            FINISHED = threadStatusQ.get()
            if FINISHED:
                break
        playsound.playsound(path)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def checkEyeStatus(landmarks):
    leftEye = [(landmarks[i][0], landmarks[i][1]) for i in leftEyeIndex]
    rightEye = [(landmarks[i][0], landmarks[i][1]) for i in rightEyeIndex]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0

    return 1 if ear >= thresh else 0  # 1 -> Open, 0 -> Closed


capture = cv2.VideoCapture(0)

# Calibration
totalTime = 0.0
validFrames = 0
dummyFrames = 100
print("Calibration in Progress!")
while validFrames < dummyFrames:
    validFrames += 1
    t = time.time()
    ret, frame = capture.read()
    height, width = frame.shape[:2]
    IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
    frame = cv2.resize(frame, None, fx=1 / IMAGE_RESIZE, fy=1 / IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)

    adjusted = histogram_equalization(frame)
    totalTime += time.time() - t

spf = totalTime / dummyFrames
print("Calibration Complete!")
print(f"Current SPF (seconds per frame): {spf:.2f} ms")

drowsyLimit = drowsyTime / spf
falseBlinkLimit = blinkTime / spf
print(f"Drowsy limit: {drowsyLimit}, False blink limit: {falseBlinkLimit}")

# Main loop
if __name__ == "__main__":
    blinkCount = 0
    drowsy = 0
    state = 0
    vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (int(capture.get(3)), int(capture.get(4))))

    while True:
        try:
            ret, frame = capture.read()
            if not ret:
                break
            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(frame, None, fx=1 / IMAGE_RESIZE, fy=1 / IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with Mediapipe
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]

                    eyeStatus = checkEyeStatus(landmarks)

                    # Blink/drowsiness logic
                    if state >= 0 and state <= falseBlinkLimit:
                        if eyeStatus:
                            state = 0
                        else:
                            state += 1
                    elif state >= falseBlinkLimit and state < drowsyLimit:
                        if eyeStatus:
                            blinkCount += 1
                            state = 0
                        else:
                            state += 1
                    else:
                        if eyeStatus:
                            state = 0
                            drowsy = 1
                            blinkCount += 1
                        else:
                            drowsy = 1

                    # Draw landmarks and check drowsiness
                    for i in leftEyeIndex + rightEyeIndex:
                        cv2.circle(frame, landmarks[i], 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

                    if drowsy:
                        cv2.putText(frame, "! ! ! DROWSINESS ALERT ! ! !", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        if not ALARM_ON:
                            ALARM_ON = True
                            threadStatusQ.put(not ALARM_ON)
                            thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
                            thread.setDaemon(True)
                            thread.start()
                    else:
                        cv2.putText(frame, f"Blinks: {blinkCount}", (460, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        ALARM_ON = False

            else:
                cv2.putText(frame, "Face not detected. Check lighting conditions.", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("Blink Detection Demo", frame)
            vid_writer.write(frame)

            k = cv2.waitKey(1)
            if k == ord('r'):
                state = 0
                drowsy = 0
                ALARM_ON = False
                threadStatusQ.put(not ALARM_ON)
            elif k == 27:
                break

        except Exception as e:
            print(e)

    capture.release()
    vid_writer.release()
    cv2.destroyAllWindows()
