import gym

env1 = gym.make('Pong-ram-v0')
env2 = gym.make('Pong-ram-v4')
env11 = gym.make('Pong-ramDeterministic-v0')
env22 = gym.make('Pong-ramDeterministic-v4')
env111 = gym.make('Pong-ramNoFrameskip-v0')
env222 = gym.make('Pong-ramNoFrameskip-v4')

def running(envs, seed=1):
    for env in envs:
        env.reset()
        env.seed(seed)
    a = [2,3]
    i = 0
    while True:
        for env in envs:
            env.render()
            o,r,d,_ = env.step(a[i])
            if d:
                env.reset()
        i = i + 1
        i = i % len(a)


envs = [env1,env11,env111,env2,env22,env222]

running(envs)

