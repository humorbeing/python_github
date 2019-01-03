import threading
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def showimg(*a):
    cv2.imshow('img', a)
    cv2.waitKey(5)
    cv2.destroyAllWindows()

def main():
    img = cv2.imread('tree.jpg')
    t = threading.Thread(target = showimg, args = (img) )
    t.start()

if __name__ == '__main__':
    main()
