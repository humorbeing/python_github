import torch
import numpy as np
from pyglet.window import key
from gym.envs.box2d.car_racing import CarRacing
from to_import import state_to_1_batch_tensor
from to_import import one_batch_tensor_to_img
from to_import import show_state

path = '/media/ray/SSD/workspace/python/dataset/save_here/carracing/v2/model/'
vae_name = 'vae_model.save'
V = torch.load(path + vae_name)
V = V.cpu()
# V = V.cuda()
V.train(False)

a = np.array([0.0, 0.0, 0.0])


def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0
    if k == key.UP:    a[1] = 0
    if k == key.DOWN:  a[2] = 0

env = CarRacing()
env.render()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release

while True:
    env.seed(seed=5)
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    env.render()
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 200 == 0 or done:
            # print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            # print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            pass
        steps += 1
        env.render()
        obs = state_to_1_batch_tensor(s)
        rec, _, _, _ = V(obs)
        img = rec.detach().numpy()
        img = one_batch_tensor_to_img(img)
        show_state(img)
        if done or restart: break
# env.monitor.close()