import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math
import screen_brightness_control as sbc

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

curr_time = 0
prev_time = 0

curr_dist = 0
prev_dist = 0
brightness_level = 50

def resizeFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

capture = cv.VideoCapture(0)
hands = mp_hands.Hands()

while True:
    isTrue, frame = capture.read()

    if (not isTrue):
        break

    frame = resizeFrame(frame, scale=1)
    frame = cv.flip(frame, 1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    if (results.multi_hand_landmarks):
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[8]
            thumb_finger_tip = hand_landmarks.landmark[4]

            h, w, _ = frame.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            thumb_x, thumb_y = int(thumb_finger_tip.x * w), int(thumb_finger_tip.y * h)

            cv.line(frame, (index_x, index_y), (thumb_x, thumb_y), (255, 0, 0), 3)
            prev_dist = curr_dist
            curr_dist = math.dist((index_x, index_y), (thumb_x, thumb_y))

            if (prev_dist > curr_dist):
                brightness_level = brightness_level - 10
                if (brightness_level < 0):
                    brightness_level = 0
                sbc.set_brightness(brightness_level)
            elif (prev_dist < curr_dist):
                brightness_level = brightness_level + 10
                if (brightness_level > 100):
                    brightness_level = 100
                sbc.set_brightness(brightness_level)
            else:
                sbc.set_brightness(brightness_level)

    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    cv.putText(frame, "FPS: {}".format(str(int(fps))), (20,40), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
    
    cv.imshow("Live Webcam", frame)
    if (cv.waitKey(1) & 0xFF==ord(' ')):
        capture.release()
        break

cv.destroyAllWindows()