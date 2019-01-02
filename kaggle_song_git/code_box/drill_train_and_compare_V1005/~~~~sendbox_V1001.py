na = [
    ['XXXXXXXXXXXXXXXXXXXXXXX', 6666666666],
    ['xxxxxxxxxx', 55555555555],
]

for i in na:
    name = i[0]+':  '
    name = name.rjust(40)
    name = name + str(i[1])
    print(name)