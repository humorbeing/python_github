from PIL import ImageGrab
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
lowerbody_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
platenum_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
upperbody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

while(True):
    imggrab = ImageGrab.grab(bbox=(200,200,800,600)) #bbox specifies specific region (bbox= x,y,width,height)
    toimg = np.array(imggrab)
    img = cv2.cvtColor(toimg, 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    upperbodies = upperbody_cascade.detectMultiScale(gray, 1.5,15)
    for (x,y,w,h) in upperbodies:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0), 1)
    '''
    platenums = platenum_cascade.detectMultiScale(gray, 1.1,10)
    for (x,y,w,h) in platenums:
        cv2.rectangle(img, (x,y), (x+w,y+h),(150,110,255), 2)


    lowerbodies = lowerbody_cascade.detectMultiScale(gray, 1.1,10)
    for (x,y,w,h) in lowerbodies:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255), 2)

    '''
    fullbodies = fullbody_cascade.detectMultiScale(gray, 1.5,15)
    for (x,y,w,h) in fullbodies:
        cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,255), 2)


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.1,10)
        for (x,y,w,h) in smiles:
            cv2.rectangle(roi_color, (x,y), (x+w,y+h),(255,255,255), 2)

    cv2.imshow("What is D looking at", img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
