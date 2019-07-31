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

def show_input(a,b='_',c='_',d='_',e='_'):
    print('| a is {}, b is {}, c is {}, '
          'd is {}, e is {}'.format(a,b,c,d,e))

print('1', end=' ')
show_input('A')
list_ = [1,2]
print(list_)
print('{}'.format(*list_))
print(*list_)
print('2', end=' ')
show_input(list_)
print('3', end=' ')
show_input(*list_)
dict_ = {
    'e': 'X',
    'd': 'Y'
}
print(dict_)
print(*dict_)
# print(**dict_)  # Error
# value = **dict_  # Error
# print(dict_[**dict_])  # Error
print('4', end=' ')
show_input(dict_)
print('5', end=' ')
show_input(*dict_)
print('6', end=' ')
show_input(*list_, **dict_)


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