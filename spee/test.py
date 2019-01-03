a = []
D = []
for i in range(2):
    a.append(0)
for i in range(8):
    D.append(a)

x=50
y=50


D[2] = [x+1,y] #E
D[6] = [x-1,y] #W
D[0] = [x,y-1] #N
D[4] = [x,y+1] #S
D[1] = [x+1,y-1] #NE
D[3] = [x+1,y+1] #SE
D[5] = [x-1,y+1] #SW
D[7] = [x-1,y-1] #NW
