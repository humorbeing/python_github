

with open('uu.txt', 'r') as f:
    a = f.readlines()

# print(len(a))
num = []
for i in a:
    if i[0] == '0':
        if len(i) == 8:
            num.append(i[:-1])
    # print(type(i[0]))
    # print('< > '*5)


# the_num = []
# for i in num:
#     print(i)
#
#     print('-'*20)

# print(len(num))
publ = []
pril = []
for i in range(61):
    j = 60 - i
    pril.append(num[j*2])
    publ.append(num[j*2+1])

# reversed(publ)
# reversed(pril)

for i in range(61):
    print('({},{})'.format(i+1, publ[i]), end='')
    # print('-'*20)
print()
for i in range(61):
    print('({},{})'.format(i+1, pril[i]), end='')
    # print('-'*20)

