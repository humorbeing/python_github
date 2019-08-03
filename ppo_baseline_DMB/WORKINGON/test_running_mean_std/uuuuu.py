import numpy as np

x = [5,6]
x = np.array(x)
print(np.mean(x))
print(np.var(x))
print(x.size)

class rms():
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 1e-4

    def update(self, x):
        x = np.array(x)
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(a_mean, a_var, a_count, b_mean, b_var, b_count):
    delta = b_mean - a_mean
    total_count = a_count + b_count

    new_mean = a_mean + delta * b_count / total_count