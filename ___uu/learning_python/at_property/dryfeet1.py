class Privatematter:
    def __init__(self):
        self.x = 1
        self._x = 2
        self.__x = 3
        y = 4
    def show_private(self):
        return self.__x

p = Privatematter()

print('p.x:', p.x)
print('p._x:', p._x)
# print('p.__x:', p.__x)  # raise AttributeError
print('p.show_private():', p.show_private())
print('p.__dict__:', p.__dict__)
print('p._Privatematter__x:', p._Privatematter__x)
# print(p.y)