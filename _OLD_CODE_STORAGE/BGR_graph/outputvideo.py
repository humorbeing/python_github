from PIL import ImageGrab
import numpy as np
import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (800,600))

while(True):
    imggrab = ImageGrab.grab(bbox=(200,200,800,600)) #bbox specifies specific region (bbox= x,y,width,height)
    toimg = np.array(imggrab)
    img = cv2.cvtColor(toimg, 0)
    out.write(toimg)
    cv2.imshow("What is D looking at", img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
out.release()
cv2.destroyAllWindows()
