import threading
import cv2

def showimg(*a):
    cv2.imshow('img', a)
    cv2.waitKey(5)
    cv2.destroyAllWindows()

def main():
    img = cv2.imread('tree.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    t = threading.Thread(target = showimg, args = (gray) )
    t.start()

if __name__ == '__main__':
    main()
