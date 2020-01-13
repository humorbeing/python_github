import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

while True:
    _, frame = cap.read() # "_" is value we don't care.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv hue sat value
    lower_red = np.array([150,150,50])
    upper_red = np.array([180,255,180]) 

    '''
    dark_red = np.uint8([[12,22,121]])
    dark_red = cv2.cvtColor(dark_red,cv2.COLOR_BGR2HSV)
    '''#no idea what this is.

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', res)

    k = cv2.waitKey(5) & 0xFF
    if k ==27: #ESC?
        break

cv2.destroyAllWindows()
cap.release()
