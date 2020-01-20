import numpy as np
import gym

RENDER = False
# RENDER = True
H = 200
D = 80*80
GAMMA = 0.99
LEARNING_RATE = 1e-4

action_map = {
    0: 1,  # stay still
    1: 2,  # up
    2: 3,  # down
}
# model = np.load('m.npy')
# print(model['W2'])
model = {}
# model['W1'] = np.random.randn(D, H) / np.sqrt(D)  # "Xavier" initialization
# model['W2'] = np.random.randn(H, 3) / np.sqrt(H)
model['W1'] = np.load('/media/ray/SSD/workspace/python/reinforcement_learning/policy-gradient/w1.npy')  # "Xavier" initialization
model['W2'] = np.load('/media/ray/SSD/workspace/python/reinforcement_learning/policy-gradient/w2.npy')
def policy(state):
    state = state[None, :]
    h = np.dot(state, model['W1'])
    h[h<0] = 0
    z = np.dot(h, model['W2'])
    exp = np.exp(z)
    return exp / np.sum(exp), h

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def gradient(dlog, h, s):
    dlog = dlog[None, :]
    s = s[None, :]
    dW2 = np.dot(h.T, dlog)
    dx = np.dot(dlog, model['W2'].T)
    dx[h==0] = 0
    dW1 = np.dot(s.T, dx)
    return dW1, dW2

when_save = 2000
env = gym.make("Pong-v0")
done = True
count = 0
while True:
    for _ in range(when_save):
        if done:
            state = env.reset()

        gw1s = []
        gw2s = []
        rewards = []
        score = 0
        while True:
            if RENDER:
                env.render()
            s = prepro(state)

            probs, h = policy(s)
            sample_number = np.random.choice(3, p=probs[0])
            action = action_map[sample_number]

            dsoftmax = softmax_grad(probs)
            dsoftmax = dsoftmax[sample_number, :]
            action_prob = probs[0, sample_number]
            dlog = dsoftmax / action_prob
            gw1, gw2 = gradient(dlog, h, s)
            state, reward, done, _ = env.step(action)
            gw1s.append(gw1)
            gw2s.append(gw2)
            rewards.append(reward)
            score += reward

            if reward == -1.0:
                break
            if reward == 1.0:
                break

        for t in range(len(gw1s)):
            g = 0
            for i, reward in enumerate(rewards[t:]):
                g += GAMMA ** i * reward

            dw1 = gw1s[t] * g
            model['W1'] = model['W1'] + LEARNING_RATE * dw1
            dw2 = gw2s[t] * g
            model['W2'] = model['W2'] + LEARNING_RATE * dw2

        if score == 1.0:
            count += 1
            print('yeah !!!!!!', count)
        else:
            count = 0
        if count == 15:
            RENDER = True

    print('saving...')
    np.save('w1.npy', model['W1'])
    np.save('w2.npy', model['W2'])
