import cv2
import numpy as np

#since I don't have camera, I will come back for this later.
cap = cv2.VideoCapture('video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,630))
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()

    cv2.imshow("frame", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    #out.write(gray)
    out.write(frame)#不能用gray，分 辨率得一样
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows() #destroyAllWindows(),don't miss 's'
