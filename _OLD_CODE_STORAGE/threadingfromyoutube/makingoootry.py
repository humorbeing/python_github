import cv2
import threading
import time

startx = 50
starty = 50
def eating(x,y,type):



def breed():



def main():
    img = cv2.imread('tree.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    playground = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for row in range(len(playground)):
        for column in range(len(playground[0])):
            if playground[row][column]>150:
                playground[row][column] = 255
            else:
                playground[row][column] = 0

    forshow = cv2.cvtColor(playground, cv2.COLOR_GRAY2BGR)




    cv2.imshow('playground',playground)
    while True:
        cv2.imshow("What's happenning",forshow)
        kk = cv2.waitKey(30) & 0xFF
        if kk ==27: #ESC?
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
