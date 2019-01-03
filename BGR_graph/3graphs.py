import threading
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

cap = cv2.VideoCapture(0)
frame = []
#frame = cv2.imread('tree.jpg') #test
ymax = 7500
lock = threading.Lock()
r = []
b = []
g = []
l = []
for i in range(256):

    b.append(0)
    g.append(0)
    r.append(0)
    l.append(i)

def getpix():
    global r
    global b
    global g
    while True:

        time.sleep(1)
        with lock:
            for i in range(256):

                b[i] = 0
                g[i] = 0
                r[i] = 0
            for pixx in frame:
                for pix in pixx:
                    b[pix[0]] += 1
                    g[pix[1]] += 1
                    r[pix[2]] += 1


def draww():
    fig = plt.figure()
    axes = plt.gca()
    axes.set_ylim([0,ymax])
    ax = fig.add_subplot(111)
    line1, = ax.plot(l,b,'b-')

    fig = plt.figure()
    axes = plt.gca()
    axes.set_ylim([0,ymax])
    ax = fig.add_subplot(111)
    line2, = ax.plot(l,g,'g-')

    fig = plt.figure()
    axes = plt.gca()
    axes.set_ylim([0,ymax])
    ax = fig.add_subplot(111)
    line3, = ax.plot(l,r,'r-')

    while True:
        with lock:
            line1.set_ydata(b)
            line2.set_ydata(g)
            line3.set_ydata(r)
            fig.canvas.draw()
            #plt.draw()
            plt.pause(0.01)

'''
def draww():
    while True:
        plt.plot(l,b)
        plt.draw()
        plt.pause(0.03)'''

def videoshow():
    global frame
    while True:
        _, img = cap.read()
        frame = img
        cv2.imshow('frame',frame)
        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()



def main():
    t1 = threading.Thread(target = videoshow)
    t2 = threading.Thread(target = draww)
    t3 = threading.Thread(target = getpix)
    #t2b = threading.Thread(target = drawone, args = (b, '-b'))
    #t2g = threading.Thread(target = drawone, args = (g, '-g'))
    #t2r = threading.Thread(target = drawone, args = (r, '-r'))

    t2.daemon = True
    t3.daemon = True
    #t2b.daemon = True
    #t2g.daemon = True
    #t2r.daemon = True

    t1.start()
    t2.start()
    #t2b.start()
    #t2g.start()
    #t2r.start()
    t3.start()


if __name__ == "__main__":
    main()
