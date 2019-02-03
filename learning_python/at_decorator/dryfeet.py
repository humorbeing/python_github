# def de(fn):
#     def cc():
#         fn()
#         print('hi')
#     return cc
# @de # turn on and off
# def dee():
#     print('no')
#
# dee()


def hi(there):
    def hi_there():
        print('Hi')
        there()
    return hi_there
@hi  # with or without
def name():
    print('Doe')

name()

def hi(there):
    def hi_there():
        print('Hi')
        there()
    return hi_there

def new_name():
    print('Jane')

same = hi(new_name)

same()