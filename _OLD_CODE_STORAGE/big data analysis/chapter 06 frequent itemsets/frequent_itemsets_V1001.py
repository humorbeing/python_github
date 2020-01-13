import numpy as np
# exer 6.1.1
baskets = list()
for b in range(1, 101):
    basket = list()
    for i in range(1, 101):
        if b%i == 0:
            basket.append(i)
    baskets.append(basket)

items = {str(i): 0 for i in range(1, 101)}
print(items)
for basket in baskets:
    for item in basket:
        items[str(item)] += 1
print(items)
for item in items:
    if items[item] >= 5:
        print("{}:{}".format(item, items[item]))

pairs ={str(i)+','+str(j): 0 for i in range(1, 101) for j in range(1, 101)}
print(pairs)
for basket in baskets:
    for i in range(1, 101):
        for j in range(1, 101):
            if i != j:
                if i in basket and j in basket:
                    pairs[str(i)+','+str(j)] += 1

print(pairs)

for i in range(1, 101):
    for j in range(1, 101):
        if pairs[str(i) + ',' + str(j)] >= 5:
            print('[{},{}]:{}'.format(i, j, pairs[str(i) + ',' + str(j)]))

sum_size = 0
for basket in baskets:
    sum_size += len(basket)

print(sum_size)