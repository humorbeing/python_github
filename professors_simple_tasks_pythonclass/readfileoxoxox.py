
infile = 'O X O X X O O X O O\nO X 5 X O O O X O O\n'
with open('grade.inp','w') as f:
    f.write(infile)

readline = open('grade.inp','r').readlines()
ar = []
for strin in readline:
    for char in strin:
        if char != ' ' and char !='\n':
            if char == 'X' or char == 'O':
                ar.append(char)
            else:
                ar.append('N')
print(ar)
scor = 0
tot = int(len(ar)/2)
for i in range(int(len(ar)/2)):
    if ar[i+tot] == ar[i]:
        scor += 2
    elif ar[i+tot] == 'N':
        pass
    else:
        scor -= 1
with open('grade.out','w') as f:
    f.write(str(scor))
