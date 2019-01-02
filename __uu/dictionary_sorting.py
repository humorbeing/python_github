import operator
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_x = sorted(x.items(), key=operator.itemgetter(1))
# reversed(sorted_x)
print(sorted_x)
for i in sorted_x:
    print(i)
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_x = sorted(x.items(), key=operator.itemgetter(0))
print(sorted_x)

stats = {'a':1000, 'b':3000, 'c': 100, 'd':3000}
m = max(stats, key=stats.get)
print(m)