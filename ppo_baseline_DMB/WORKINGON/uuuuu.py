import gym

# env = gym.make('Pong-ram-v0')
# env = env.env
# s = env.reset()
# while True:
#     env.render()
#     s, r, d, i = env.step(env.action_space.sample())
#     if r != 0:
#         print(r)
#     if d:
#         env.reset()


# print( '5 / 2 =',5 /2)
# print( '5 // 2 =',5 // 2)
# print( '5 % 2 =',5 % 2)
# print( '5 / 3 =',5 / 3)
# print( '5 // 3 =',5 //3)
# print( '5 % 3 =',5 % 3)
# import math
# print( 'math.ceil(5 / 4) = ', math.ceil(5/4))

# def show_input(a,b='_',c='_',d='_',e='_'):
#     print('| a is {}, b is {}, c is {}, '
#           'd is {}, e is {}'.format(a,b,c,d,e))
#
# print('1', end=' ')
# show_input('A')
# list_ = [1,2]
# print(list_)
# print('{}'.format(*list_))
# print(*list_)
# print('2', end=' ')
# show_input(list_)
# print('3', end=' ')
# show_input(*list_)
# dict_ = {
#     'e': 'X',
#     'd': 'Y'
# }
# print(dict_)
# print(*dict_)
# # print(**dict_)  # Error
# # value = **dict_  # Error
# # print(dict_[**dict_])  # Error
# print('4', end=' ')
# show_input(dict_)
# print('5', end=' ')
# show_input(*dict_)
# print('6', end=' ')
# show_input(*list_, **dict_)
#

"""
1 | a is A, b is _, c is _, d is _, e is _
[1, 2]
1
1 2
2 | a is [1, 2], b is _, c is _, d is _, e is _
3 | a is 1, b is 2, c is _, d is _, e is _
{'e': 'X', 'd': 'Y'}
e d
4 | a is {'e': 'X', 'd': 'Y'}, b is _, c is _, d is _, e is _
5 | a is e, b is d, c is _, d is _, e is _
6 | a is 1, b is 2, c is _, d is Y, e is X
"""




#
# def aaa(a,b=0,c=0,d=0,e=0):
#     print(a,b,c,d,e)
#
# aaa(1)
#
# def bbb(*a, **b):
#     print(a)
#     print(b)
#     aaa(*a,**b)
# def zzz(a,b):
#     aaa(*a,**b)
#
#
# bbb(1,2,d=1,e=1)
# ccc = [2,3]
# eee = {'e':5}
# bbb(*ccc,**eee)
# bbb(2,5,**eee)
#
# zzz([1,2],eee)
# zzz([1,2,3],eee)


# def show_input(a,b='_',c='_',d='_',e='_'):
#     print('| a is {}, b is {}, c is {}, '
#           'd is {}, e is {}'.format(a,b,c,d,e))
#
# def deliver_value(*list_, **dict_):  # Made up op name
#     print('In D, list is',list_)
#     print('In D, dictionary is', dict_)
#     show_input(*list_,**dict_)
#
# print('1-------------------------------')
# deliver_value('A','B',e='E')
# print()
#
# print('2-------------------------------')
# list_ = ['A','B']
# deliver_value(*list_)
# deliver_value(list_)
# print()
#
# print('3-------------------------------')
# dict_ = {
#     'e': 'E',
#     'd': 'D'
# }
# deliver_value('A','B',**dict_)
# deliver_value('A','B',dict_)
# deliver_value(*list_,**dict_)
# print()
#
# def pass_value(list_,dict_):  # Made up op name
#     show_input(*list_,**dict_)
#
# print('4-------------------------------')
# pass_value('A', dict_)
# # pass_value('A','B', dict_)   # Error
# # deliver_value('A','B',dict_) # No error
# # pass_value('A')              # Error
# # deliver_value('A')           # No error
# pass_value('A', {'e':'E'})

# if isinstance(venv, VecNormalize):


import gym
# name = 'AirRaid-ramNoFrameskip-v4'
# name = 'Alien-ramDeterministic-v4'
# name = 'Amidar-ramDeterministic-v4'
# name = 'Amidar-ramNoFrameskip-v4'
# name = 'Assault-ramNoFrameskip-v4'
# name = 'Assault-ramDeterministic-v4'
# name = 'Atlantis-ramDeterministic-v4'
# name = 'Bowling-ramDeterministic-v4'
# name = 'Breakout-ramDeterministic-v4'
# name = 'Pitfall-ramNoFrameskip-v4'
# name = 'Phoenix-ramNoFrameskip-v4'
# name = 'Pong-ram-v4'
# name = 'PongNoFrameskip-v4'
# # name = 'AssaultNoFrameskip-v4'
# # env1 = gym.make(name)
#
# def mk(seed, name=name):
#     env = gym.make(name)
#     env.seed(seed)
#     return env
# num_env = 5
# seed = 10
# envs = []
# for i in range(num_env):
#     envs.append(mk(seed+i))  # Diff seed
#     # envs.append(mk(seed))  # Same seed
# for env in envs:
#     env.reset()
# while True:
#     for env in envs:
#         env.render()
#         _,_,d,_ = env.step(1)
#         if d:
#             env.reset()
# env = env.env

# env1.seed(400)
# env.seed(1)
# env1.reset()
# env.reset()
# while True:
#     env1.render()
#     env.render()
#     o, r, d, i = env.step(1)
#
#     if d:
#         env.reset()
#     o, r, d, i = env1.step(1)
#
#     if d:
#         env1.reset()

print(3e7)