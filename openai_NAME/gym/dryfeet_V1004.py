from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
for i in range(10):
    x = space.sample()
    print(x)
    assert space.contains(x)
assert space.n == 8