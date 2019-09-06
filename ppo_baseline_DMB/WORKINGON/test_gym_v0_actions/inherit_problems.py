
class CC():
    def __init__(self):
        self.ccc = 0.1


class A():
    def __init__(self, cc):
        self.cc = cc
        self.a1 = 1
        self.a2 = 2

class B(A):
    def __init__(self, cc):
        super().__init__(cc)  # this is python 3, following 2 line have same effect
        # super(B, self).__init__()  # in python 2
        # A.__init__()  # don't do this
        self.b1 = 10
        self.b2 = 20

class C(B):
    def __init__(self, cc):
        super().__init__(cc)
        self.c1 = 100
        self.c2 = 200

class D(A):
    def __init__(self, cc):
        super().__init__(cc)
        self.d1 = 1000

o = CC()
print(o.ccc)
o = C(o)
print(o.b1)
o = D(o)
print(o.a1)

# print(o.cc.ccc)  # fix is below
print(o.cc.cc.ccc)

# print(o.b1)  # fix is below
print(o.cc.b1)
# print(o.c1)
# print(dddd)
#
# dddd = 1

class Animal():
    def __init__(self):
        self.live_on = 'Earth'

class Human(Animal):
    # def __init__(self, me):
    def __init__(self):
        super().__init__()
        # self.me = me
        self.finger = 10

class GrandFather(Human):
    # def __init__(self, grandson):
    #     super().__init__(grandson)
    def __init__(self):
        super().__init__()
        self.name = 'M'
        self.family_name = 'John'

class Father(GrandFather):
    # def __init__(self, son):
    #     super().__init__(son)

    def __init__(self):
        super().__init__()
        self.name = 'O'


class Mother(Human):
    # def __init__(self, baby):
    #     super().__init__(baby)
    def __init__(self):
        super().__init__()
        self.name = 'K'

baby = Animal()
son = Father()
me = son
# me = Mother()
print('- = + '*20)
print(me.name)
print(me.family_name)
print(me.finger)
print(me.live_on)

# print(me.family_name)
# print(me.me.family_name)
#
# print(me.finger)
#
# # print(me.live_on)
# print(me.me.me.live_on)


class Me():
    def __init__(self):
        self.first_name = 'Michael'
        self.like = 'I like playing computer games'

class Human():
    def __init__(self, input_person):
        self.me = input_person
        self.fingers = 10
    def hi(self):
        print('hi world')

class FromFather(Human):
    def __init__(self, son):
        super().__init__(son)
        self.family_name = 'Jackson'
        self.eye = 'Blue'
    def hi(self):
        print('hi dad')

class FromMother(Human):
    def __init__(self, me):
        super().__init__(me)
        self.hair = 'Yellow'
    def hi(self):
        print('hi mom')

me = Me()

print(' ||||  || |||   '*10)
print()
print(me.first_name)
print(me.like)

me = Human(me)

print(me.fingers)
print(me.me.first_name)
print(me.me.like)

me = FromFather(me)

# me = FromFather()
# print(me.name)
print('My eye is ' + me.eye)
print(me.family_name)
print(me.fingers)
print(me.me.fingers)
me.hi()

me = FromMother(me)
me.hi()
#
#
# me = FromMother()
# print(me.name)

print(' NEW '*30)
print()

class House():
    def __init__(self):
        self.name = 'Big house'
    def show(self):
        pass

class Decorate():
    def __init__(self, house):
        self.house = house
    def show(self):
        pass

class TV(Decorate):
    def __init__(self, house):
        super().__init__(house)  # python3 style
        # super(TV, self).__init__(house)  # python2 style
        # Decorate.__init__(house)  # Avoid this
        self.tv_size = '60 inch'
    def show(self):
        self.house.show()
        print('Add a tv, check tv_size')

class Fridge(Decorate):
    def __init__(self, house):
        super(Fridge, self).__init__(house)
        self.fridge_size = '180 cm'
    def show(self):
        self.house.show()
        print('Add a Fridge, check fridge_size')

class Chair(Decorate):
    def __init__(self, house):
        super().__init__(house)
        self.chair_color = 'Black'

    def show(self):
        self.house.show()
        print('Add a Chair, check chair_color')

house = House()
print('Welcome to', house.name)

house = TV(house)
house = Fridge(house)
house = Chair(house)
house.show()

def check_detail(house, what):
    if isinstance(house, what):
        return house
    elif hasattr(house, 'house'):
        return check_detail(house.house, what)

tv = check_detail(house, TV)
print('TV size is', tv.tv_size)
fridge = check_detail(house, Fridge)
print('Fridge size is', fridge.fridge_size)
chair = check_detail(house, Chair)
print('Chair color is', chair.chair_color)
name = check_detail(house, House)
print(house.chair_color, 'Chair in the', name.name)

print()
print('------- SHOW DETAILS -------')
#show class detail
# __dict__ equivalent to vars()
# does not show build in functions
# print(house.__dict__.keys())  # same as vars()
print('vars():',vars(house).keys())
# print(vars(house).items())  # show items from dictionary

# dir() shows EVERYTHING
# which includes functions, where vars() doesn't
attributes = [attr for attr in dir(house)
              if not attr.startswith('__')]
print('dir():',attributes)
print('EVERYTHING from dir():',dir(house))

print()
print('------- SHOW What CLASS -------')

print('house __class__:', house.__class__)
print('is house TV class:',isinstance(house, TV))
print()
house.__class__ = TV
print('house __class__:', house.__class__)
print('is house TV class:',isinstance(house, TV))
