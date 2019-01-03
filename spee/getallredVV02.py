#fixed cross colorring.
#to fix. mama not giving birth and go rest.

import cv2
import threading
import time

global X,Y,BIG
X = 0
Y = 0
BIG = [[]]
startx = 50
starty = 50

maxthread = 100
sle = 0.02



def scout(y,x,cell_type):

    global breedground,growth,X,Y,BIG

    D = [[0 for i in range(2)] for j in range(8)]
    issafe = [-1 for i in range(8)]



    if cell_type == -1:
        celltype = breedground[y][x]
        BIG[y][x] = 1
    else:
        celltype = cell_type

    D[2] = [y,x+1] #E
    D[6] = [y,x-1] #W
    D[0] = [y-1,x] #N
    D[4] = [y+1,x] #S
    D[1] = [y-1,x+1] #NE
    D[3] = [y+1,x+1] #SE
    D[5] = [y+1,x-1] #SW
    D[7] = [y-1,x-1] #NW
    if y == 0:
        issafe[0]=0
        issafe[1]=0
        issafe[7]=0
    if y == Y-1:
        issafe[4]=0
        issafe[5]=0
        issafe[3]=0
    if x == 0:
        issafe[5]=0
        issafe[6]=0
        issafe[7]=0
    if x == X-1:
        issafe[1]=0
        issafe[2]=0
        issafe[3]=0

    for i in range(8):
        #if D[i][0] < 0 | D[i][0] > Y-1 | D[i][1] < 0 | D[i][1] > X-1:
        #    issafe[i] = 0 #edge
        #elif BIG[D[i][0]][D[i][1]] == 1:
        if issafe[i] != 0:
            if BIG[D[i][0]][D[i][1]] == 1:
                issafe[i] = 5
            else:
                issafe[i] = 1 #not edge

    for i in range(8):
        if issafe[i] == 1:
            if breedground[D[i][0],D[i][1]] == celltype:
                issafe[i] = 2 #good to breed
            else:
                issafe[i] = 3 #not a favor ground

    for i in range(8):
        if issafe[i] == 2:
            time.sleep(sle)
            replication(D[i][0],D[i][1],celltype)
            issafe[i] = 4 #breeded






def replication(y,x,celltype):

    global growth,BIG
    if threading.active_count() < maxthread:
        growth[y][x] = [0,0,255]
        t = threading.Thread(target = scout, args = (y,x,celltype))
        t.daemon = True
        t.start()
        BIG[y][x] = 1
    else:
        pass

def mon():
    while True:

        #print("thread count is {}".format(threading.active_count()))
        pass
def main():
    global breedground,growth,X,Y,BIG
    img = cv2.imread('tree.jpg')
    #img = cv2.imread('car.jpg')
    #img = cv2.imread('mat.jpg')
    #img = cv2.imread('image.jpg')
    Y = len(img)
    X = len(img[0])
    BIG = [[0 for i in range(X)] for j in range(Y)]


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    breedground = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for row in range(len(breedground)):
        for column in range(len(breedground[0])):
            if breedground[row][column]>150:
                breedground[row][column] = 255
            else:
                breedground[row][column] = 0



    growth = cv2.cvtColor(breedground, cv2.COLOR_GRAY2BGR)

    t = threading.Thread(target = scout, args = (starty,startx,-1))
    t.start()
    moniter = threading.Thread(target = mon)
    moniter.daemon = True
    moniter.start()


    cv2.imshow('breedground',breedground)
    while True:
        cv2.imshow('growth',growth)

        kk = cv2.waitKey(30) & 0xFF
        if kk ==27: #ESC?
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
