import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


cap = cv2.VideoCapture(0)

while True:
    bleh, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    plt.subplot(111)
    plt.imshow(img,cmap = 'gray')
    plt.title('Original')
    plt.draw()
    time.sleep(0.2)
