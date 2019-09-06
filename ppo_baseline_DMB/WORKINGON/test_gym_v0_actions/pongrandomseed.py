import gym

# env1 = gym.make('Pong-ram-v0')
# env2 = gym.make('Pong-ram-v4')
# env11 = gym.make('Pong-ramDeterministic-v0')
# env22 = gym.make('Pong-ramDeterministic-v4')
# env111 = gym.make('Pong-ramNoFrameskip-v0')
# env222 = gym.make('Pong-ramNoFrameskip-v4')
env1 = gym.make('PongDeterministic-v4')#gym.make('Pong-v0')
env2 = gym.make('PongDeterministic-v4')#gym.make('Pong-v4')
env11 = gym.make('PongDeterministic-v4')#gym.make('PongDeterministic-v0')
env22 = gym.make('PongDeterministic-v4')#gym.make('PongDeterministic-v4')
env111 = gym.make('PongDeterministic-v4')#gym.make('PongNoFrameskip-v0')
env222 = gym.make('PongDeterministic-v4')#gym.make('PongNoFrameskip-v4')
env3 = gym.make('PongDeterministic-v4')
env4 = gym.make('PongDeterministic-v4')
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

def mvrunning(envs):
    seed = [1, 4, 5, 10, 100, 200, 5000, 9999]
    for n in range(len(envs)):
        envs[n].reset()
        envs[n].seed(seed[n])
    a = [2,3,3,3,3,3,3,3,2,2,3,2,3,3,2,2,3,3,3,3,3,2,2,3,2,3,3,2,2,2,2,2,2,2,2,3,3,2,2,3,3,3,3,3,2,2,3,2,3,3,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,2,2]
    i = 0
    while True:
        for n in range(len(envs)):

            envs[n].render()
            o,r,d,_ = envs[n].step(a[i])
            if d:
                envs[n].reset()
                envs[n].seed(seed[n])
        i = i + 1
        i = i % len(a)

envs = [env1,env11,env111,env2,env22,env222, env3, env4]

mvrunning(envs)

