import numpy as np


def ranking(fit_in):
    fit = [0 for j in range(len(fit_in))]
    fit_list = fit_in
    order_list = [i for i in range(len(fit_list))]
    for i in range(len(fit_list)):
        n = np.argmax(fit_list)
        l = order_list[n]
        fit[l] = i
        fit_list = np.delete(fit_list, n)
        order_list = np.delete(order_list, n)
    return np.array(fit)


a = [3, 5, 4, 5, 4,1,1,1,1,2]

print(ranking(a))