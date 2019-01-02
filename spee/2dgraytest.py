import cv2
import numpy as np

img = cv2.imread('tree.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for row in range(len(b)):
    for column in range(len(b[0])):
        if b[row][column]>150:
            b[row][column] = 255
        else:
            b[row][column] = 0

c = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

for i in range(10):
    for j in range(10):
        c[i+50][j+50]= [0,0,255]

cv2.imshow('Pic',img)
cv2.imshow('gray',gray)
cv2.imshow('bbbb',b)
cv2.imshow('cccc',c)

cv2.waitKey(0)
cv2.destroyAllWindows()
