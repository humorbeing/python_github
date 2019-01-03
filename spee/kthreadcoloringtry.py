import threading
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

#c = []
def k(*a):
    print('running')

    for i in range(50):
        for j in range(50):
            time.sleep(1)
            a[i][j] = [0,0,255]


    '''
    while True:
        a[50+i][50] = [0,0,255]
        i += 1
        if a[50+i][50] == 255 & 50+i < len(a):
            goodtopaint = True
        else:
            goodtopaint = False
            '''
    #for i in range (100):
    '''
    a[150][20] = [0,0,255]
    a[150][21] = [0,0,255]
    a[150][22] = [0,0,255]
    a[150][23] = [0,0,255]
    a[150][24] = [0,0,255]
    a[150][25] = [0,0,255]
    a[150][26] = [0,0,255]
    '''
    '''
    c[150][20] = [0,0,255]
    c[150][21] = [0,0,255]
    c[150][22] = [0,0,255]
    c[150][23] = [0,0,255]
    c[150][24] = [0,0,255]
    c[150][25] = [0,0,255]
    c[150][26] = [0,0,255]
    '''
    print("red complete")
def kk(*aa):
    print('running kk')
    bc = cv2.cvtColor(aa, cv2.COLOR_GRAY2BGR)
    t1 = threading.Thread(target = k, args = (bc) )
    t1.daemon = True
    t1.start()
    while True:
        cv2.imshow('bcbc',bc)
        kk = cv2.waitKey(5) & 0xFF
        if kk ==27: #ESC?
            break


def main():



    #global c



    img = cv2.imread('tree.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for row in range(len(b)):
        for column in range(len(b[0])):
            if b[row][column]>150:
                b[row][column] = 255
            else:
                b[row][column] = 0

    #c = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    '''
    for i in range(10):
        for j in range(10):
            c[i+50][j+50]= [0,0,255]'''
    #t1 = threading.Thread(target = k)
    t = threading.Thread(target = kk, args = (b) )
    t.daemon = True
    t.start()

    cv2.imshow('Pic',img)
    cv2.imshow('gray',gray)
    cv2.imshow('bbbb',b)
        #cv2.imshow('cccc',c)


    cv2.waitKey(5)

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
