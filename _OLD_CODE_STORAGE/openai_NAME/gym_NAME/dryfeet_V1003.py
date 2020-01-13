import gym
# import tensorflow
env = gym.make('Breakout-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])
