import random

N = 50
M = N - 2
p = 0.50


def makeS():
    global M, p
    A = [[-1 for i in range(M)] for j in range(M)]
    for i in range(M):
        for j in range(M):
            if i != j:
                if random.random() < p:
                    if A[i][j] == -1:
                        A[i][j] = int(random.random()*100)
                        A[j][i] = 0
                    else:
                        if random.random() > 0.5:
                            A[i][j] = int(random.random() * 100)
                            A[j][i] = 0
    for i in range(M):
        if any([j > 0 for j in A[i]]):
            # print(i)
            pass
        else:

            l = [j for j in range(M) if A[i][j] == -1]
            l.remove(i)
            random.shuffle(l)

            if i < int(M/2):
                chose = i + 1
            else:
                chose = i - 1
            if l:
                chose = l[0]

            A[i][chose] = int(random.random() * 100)
            A[chose][i] = 0
    for i in range(M):
        if A[i].count(0) == 0:
            l = [j for j in range(M) if A[i][j] == -1]
            l.remove(i)
            random.shuffle(l)
            if i < int(M/2):
                chose = i + 1
            else:
                chose = i - 1
            if l:
                chose = l[0]

            A[i][chose] = 0
            A[chose][i] = int(random.random() * 100)
    return A


def checkS(SS):
    global M
    good = True
    for i in range(M):
        if any([j > 0 for j in A[i]]):
            pass
        else:
            good = False
        if SS[i].count(0) > 0:
            pass
        else:
            good = False
        if SS[i].count(-1) == 0:
            good = False
    return good
A = makeS()
while not checkS(A):
    A = makeS()


a = [i for i in range(M)]
random.shuffle(a)
x = int(M*p)
y = int(M*p/2)

z = random.randint(y, x)
if z < 2:
    z = 2
iii = a[0:z]
iii = sorted(iii)

def makeooo():
    global M, p
    b = [i for i in range(M)]
    random.shuffle(b)
    x = int(M*p)
    y = int(M*p/2)

    zz = random.randint(y, x)
    if zz < 2:
        zz = 2

    return b[0:zz]
ooo = makeooo()
ooo = sorted(ooo)
while iii == ooo:
    ooo = makeooo()
    ooo = sorted(ooo)

outputstring = []
outputstring.append(str(N)+'\n')
s = '1 '+str(len(iii))
for i in iii:
    s += ' '+str(i+2)+' '+str(int(random.random()*100))
s += '\n'
outputstring.append(s)
for i in range(M):
    s = str(i+2)
    n = M - (A[i].count(-1) + A[i].count(0))
    if i in ooo:
        n += 1
    s += ' '+str(n)
    for j in range(M):
        if A[i][j] > 0:
            s += ' '+str(j+2)+' '+str(A[i][j])
    if i in ooo:
        s += ' '+str(N)+' '+str(int(random.random() * 100))
    s += '\n'
    outputstring.append(s)
pass

# firstline = "{} {} \n".format(N,M)
# def lineto(B):
#     sum = 0
#     s = ""
#     for i in range(M):
#         if B[i] == 1:
#             sum += 1
#             s += str(i+1)+' '
#     return "{} ".format(sum)+s
with open('network.inp', 'w') as f:
    f.writelines(outputstring)
    # for i in range(N):
    #     f.write("{} ".format(i+1)+lineto(A[i])+'\n')
