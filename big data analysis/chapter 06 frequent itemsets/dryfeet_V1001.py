import numpy as np


# exer 6.1.1

baskets = list()
for b in range(1, 101):
    basket = list()
    for i in range(1, 101):
        if b%i == 0:
            basket.append(i)
    baskets.append(basket)

print(baskets)

# exer 6.1.3

baskets = list()
for b in range(1, 101):
    basket = list()
    for i in range(1, 101):
        if i%b == 0:
            basket.append(i)
    baskets.append(basket)

print(baskets)

# exer 6.1.4

baskets = list()
for b in range(1, 101):
    basket = list()
    for i in range(1, 11):
        if np.random.random() < 1/i:
            basket.append(i)
    baskets.append(basket)

print(baskets)