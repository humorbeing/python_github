import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

while True:
    _, frame = cap.read() # "_" is value we don't care.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv hue sat value
    lower_red = np.array([0,200,120])
    upper_red = np.array([255,255,250])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = 1)
    dilation = cv2.dilate(mask, kernel, iterations = 1)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #false positive
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #false nagitive


    cv2.imshow('frame', frame)
    cv2.imshow('result', res)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)


    k = cv2.waitKey(5) & 0xFF
    if k ==27: #ESC?
        break

cv2.destroyAllWindows()
cap.release()
