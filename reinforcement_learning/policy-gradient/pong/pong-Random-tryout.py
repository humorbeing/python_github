import numpy as np
import gym

H = 200  # number of hidden layer neurons
D = 80 * 80  # input dimensionality: 80x80 grid
model = {}
model['W1'] = np.random.randn(H, D) / np.sqrt(D)
a = np.random.randn(2,3)
print(a)
a = a / np.sqrt(3)
print(a)
x = np.random.randn(6400)
print(x.shape)
W1 = np.random.randn(200, 6400)
h = np.dot(W1, x)
h = W1.dot(x)
# h = x.dot(W1)  # error
print('aa')
print(h.shape)
W2 = np.random.randn(200)
logp = np.dot(W2, h)
print(logp.shape)
print(logp)
prev_x = 5

cur_x = 1

x = cur_x - prev_x if prev_x is not None else 0
print(x)
if prev_x is None:
    x = 0
else:
    x = cur_x - prev_x

print(x)

print(np.random.uniform())

# env = gym.make("Pong-v0")
# env.reset()
# while True:
#     env.render()
#     action = 2
#     o,r,d,i = env.step(action)
#     print(r)

def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def pr(s):
    x = softmax_grad(s)
    print(x)

for i in range(1):
    x = np.random.rand(1, 2)
    pr(x)