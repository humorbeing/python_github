import gym

env = gym.make('Pong-ram-v0')

state = env.reset()
print(state)

while True:
    env.render()
    _,_,done,_ = env.step(3)
    if done:
        env.reset()
