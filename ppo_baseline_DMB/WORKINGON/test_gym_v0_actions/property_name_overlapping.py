class C:
    def __init__(self, x):
        self.x = x
        self.y = x
        self.hidden = x
        print('----> Inside "__init__":')
        print('---->  self.__x:' + self.__x)
        print('---->  self._C__x:' + self._C__x)
        # print('---->  self.__y:' + self.__y)  # Error
        # print('---->  ' + self._C__y)  # Error
        # print('---->  ' + self._C__z)  # Error
        # print('---->  ' + self._C__hidden)  #Error
        print('----> End of "__init__".\n')

    @property
    def x(self):
        return 'The secret is protected !!!\n' \
               '          But, I am telling you anyway.\n' \
               '          ' + self.__x
    # if use self.x, Error: maximum recursion depth exceeded.
    # When only there is "self.x", "@property def x(self)" and
    # follows with "@x.setter", Three are all answer the "self.x".
    # Then, "__x" is an MUST-use alternative way of saying "self.x",
    # to prevent Error: maximum recursion depth exceeded.

    # Can not do this with "self.y", like self__y, or c._C__y,
    # Neither from outside of class or inside of class

    # Can not do this with "self.z" or "self.hidden"

    @x.setter  # this x in x.setter must match with "def x(self, str)"'s x
    def x(self, x_in):
        self.__x = x_in

    @property
    def z(self):
        return self.hidden

    @z.setter
    def z(self, z):
        self.hidden = z

c = C('The secret is 42')
print('From c.x:', c.x)
print('From c._C__x:', c._C__x)
# print(c.__x)  # Error

print('From c.y:', c.y)
# print(c._C__y)  # Error

print('From c.z:', c.z)
# print(c._C__z)  # Error
# print(c._C__hidden)  # Error