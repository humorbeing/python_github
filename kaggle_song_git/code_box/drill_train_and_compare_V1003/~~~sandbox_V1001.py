a = []

b = ['a', 'b']
c = ['b', 'c']
d = ['a', 'd']

a.append(b)
a.append(c)
a.append(d)

if ['a', 'c'] in a or ['c', 'a'] in a:
    print('here')