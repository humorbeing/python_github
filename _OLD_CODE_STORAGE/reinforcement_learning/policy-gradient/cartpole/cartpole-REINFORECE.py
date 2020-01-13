import gym
import numpy as np

GAMMA = 0.99
LEARNING_RATE = 0.001
# RENDER = True
RENDER = False
theta = np.random.rand(4, 2)
env = gym.make('CartPole-v1')
action_map = {
    0: 0,
    1: 1
}
def policy(state, w):
    z = state.dot(w)
    exp = np.exp(z)
    return exp / np.sum(exp)

def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

max_count = 0
while True:

    state = env.reset()
    grads = []
    rewards = []

    score = 0
    count = 0
    while True:
        if RENDER:
            env.render()
        state = state[None, :]
        probs = policy(state, theta)
        sample_number = np.random.choice(2, p=probs[0])
        action = action_map[sample_number]

        dsoftmax = softmax_grad(probs)
        dsoftmax = dsoftmax[sample_number, :]
        action_prob = probs[0, sample_number]
        dlog = dsoftmax / action_prob

        dlog = dlog[None, :]
        grad = np.dot(state.T, dlog)
        state, reward, done, _ = env.step(action)

        grads.append(grad)
        rewards.append(reward)

        score += reward

        if done:
            count += 1
        if count == 100:
            break

    for t in range(len(grads)):
        g = 0
        for i, reward in enumerate(rewards[t:]):
            g += GAMMA ** i * reward

        dtheta = grads[t] * g
        theta = theta + LEARNING_RATE * dtheta
    if not RENDER:
        if score >= 500:
            max_count += 1
            print(max_count)
        if max_count == 200:
            RENDER = True


