from pyDatalog_playground import pyDatalog

def is_in(a, l):
    print(a)
    print(l)
    if a in l:
        return True
    else:
        return False

pyDatalog.create_terms('X,Y,Z')
pyDatalog.create_terms('Miss,Cann,NEXT_Miss,NEXT_Cann')
pyDatalog.create_terms('Boat,NEXT_Boat')
pyDatalog.create_terms('is_ok,is_side_ok,next_stage,from_to')
pyDatalog.create_terms('is_in')

print(X.in_((0,1,2,3,4)))

print(X.in_(range(5)).data)
print(X.in_(range(5)) == set([(0,), (1,), (2,), (3,), (4,)]))

print("Data : ",X.data)
print("First value : ",  X.v())
# is_ok(X, Y) <= (Z==True) & (Z==is_in(X, Y))
# from_to([Miss,Cann,Boat], X, Y)\
# <= ([Miss,Cann,Boat]._in(X))
#
# print(from_to([1,1,1], X))
print((Y==[[1,2]]) & (X==Y + [[2,2]]))
# print(X.data[0])
# print(X.v())
is_ok(X, Y) <= (X==5) & (Z==is_in(X, Y))
print(is_ok(Y,X))