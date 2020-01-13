import random

N = 500
M = 500
israndom = False
p = 0.03

A = [[0 for i in range(M)] for i in range(N)]
for i in range(N):
    for j in range(M):
        if israndom:
            if random.random()<random.random():
                A[i][j] = 1
        elif random.random()<p:
            A[i][j] = 1
firstline = "{} {} \n".format(N,M)
def lineto(B):
    sum = 0
    s = ""
    for i in range(M):
        if B[i] == 1:
            sum += 1
            s += str(i+1)+' '
    return "{} ".format(sum)+s
with open('matching.inp','w') as f:
    f.write(firstline)
    for i in range(N):
        f.write("{} ".format(i+1)+lineto(A[i])+'\n')
