class c:
    def __init__(self, b):
        self.x = b
    @property
    def x(self):
        print('Not showing')
        # return self.__x
    @x.setter
    def x(self, a):
        self.__x = a


d = c(10)

print(d.x)
print(d._c__x)

class C:
    def __init__(self, x):
        self.x = x
        self.y = x
    @property
    def x(self):
        return 'That is protected !!!'
    @x.setter
    def x(self, x_in):
        self.__x = x_in


c = C('Secret is 42')
print('From c.x:', c.x)
print('From c.y:', c.y)
print('From c._C__x:', c._C__x)

print('---=-=-='*20)

class C:
    def __init__(self, x):
        self.x = x
        self.y = x

    def get_x(self):
        return 'That is protected !!!'

    def set_x(self, x_in):
        self.__x = x_in

    # I think it's defining function x in class
    # like def x(self)
    # There for self.x is actually calling function.
    x = property(get_x, set_x)

    # Or
    # x = property()
    # x = x.getter(get_x)
    # x = x.setter(set_x)

c = C('Secret is 42')
print('From c.x:', c.x)
print('From c.y:', c.y)
print('From c._C__x:', c._C__x)