import cv2
import numpy as np
import matplotlib.pyplot as plt


plt.show()
cap = cv2.VideoCapture('face.mp4')
#img = cv2.imread('template.jpg')
r = []
b = []
g = []
l = []
for i in range(256):

    b.append(0)
    g.append(0)
    r.append(0)
    l.append(i)


while True:
    _, frame = cap.read()


    for i in range(256):

        b[i] = 0
        g[i] = 0
        r[i] = 0
    '''for pixx in img:
        for pix in pixx:
            print(pix)
            print(b[pix[0]])
            b[pix[0]] += 1
            #print("blue 12 is {}".format(b[12]))
            g[pix[1]] = g[pix[1]] + 1
            r[pix[2]] = r[pix[2]] + 1
    #print("---blue 12 is {}".format(b[12]))
    '''

    #print("with one {}".format(frame[0]))
    #print("with two {}".format(frame[0][0]))
    #print("with three {}".format(frame[0][0][0]))
    #print("with four {}".format(frame[0][0][0][0]))
    for pixx in frame:
        for pix in pixx:
            b[pix[0]] += 1
            g[pix[1]] += 1
            r[pix[2]] += 1
    plt.plot(l,b)
    plt.draw()
    plt.pause(0.01)
    '''for i in range(256):
        if b[i] > 200:
            print("b {} is {}".format(i,b[i]))'''
    #img = cv2.
    #cv2.imshow('img',img)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(300) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
