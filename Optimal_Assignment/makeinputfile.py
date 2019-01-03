from random import randint
N = 500
B = [' '.join(str(randint(1, 1000)) for i in range(N)) for j in range(N)]
with open('assign.txt','w') as f:
    f.write(str(N)+'\n')
    for i in B:
        f.write(i+'\n')
