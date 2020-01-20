from pyDatalog_playground import pyDatalog

def is_in(x,y):
    if x in y:
        return False
    else:
        return True

def print_detail(x,y,z):
    output_line = ''
    if x==0:
        pass
    elif abs(x)==1:
        output_line += 'One Missionary '
    else:
        output_line += 'Two Missionaries '
    if y==0:
        pass
    elif abs(y)==1:
        output_line += 'One Cannibal '
    else:
        output_line += 'Two Cannibals '
    if z==1:
        output_line += 'from LEFT to RIGHT.'
    else:
        output_line += 'from RIGHT to LEFT.'
    return output_line

def ppp(x):

    if (0, 0, 0) in x:
        print()
        print('One Solution is:')
        for i in range(len(x)-1):
            print(
                x[i], '->', x[i+1],
                ': ', print_detail(x[i][0]-x[i+1][0],
                                   x[i][1]-x[i+1][1],
                                   x[i][2]-x[i+1][2])
                )
        print('End of this Solution.')

pyDatalog.create_terms('X,Y,Z,O,P,Q')
pyDatalog.create_terms('Miss,Cann,NEXT_Miss,NEXT_Cann')
pyDatalog.create_terms('Boat,NEXT_Boat')
pyDatalog.create_terms('is_ok,is_side_ok,next_state,from_to')
pyDatalog.create_terms('ppp,is_in')
print('Missionary and Cannibal Problem.')
print('2 missionaries and 2 cannibal have a boat on left side of river.')
print('They want to cross river without anyone getting eaten.')
print('Goal is (2,2,1) -> (0,0,0)')
#2Miss L->R, [2,_,1] -> [0,_,0]
next_state([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (Miss==2) & (NEXT_Miss==0) & (NEXT_Cann==Cann)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#2Cann L->R, [_,2,1] -> [_,0,0]
next_state([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (Cann==2) & (NEXT_Miss==Miss) & (NEXT_Cann==0)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Miss and 1Cann L->R, [_,_,1] -> [-1,-1,0]
next_state([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (NEXT_Miss==Miss-1) & (NEXT_Cann==Cann-1)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Miss L->R, [_,_,1] -> [-1,_,0]
next_state([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (NEXT_Miss==Miss-1) & (NEXT_Cann==Cann)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Cann L->R, [_,_,1] -> [_,-1,0]
next_state([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (NEXT_Miss==Miss) & (NEXT_Cann==Cann-1)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

####################---------######################

#2Miss R->L, [0,_,0] -> [2,_,1]
next_state([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
<= (Miss==0) & (NEXT_Miss==2) & (NEXT_Cann==Cann)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#2Cann R->L, [_,0,0] -> [_,2,1]
next_state([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
<= (Cann==0) & (NEXT_Miss==Miss) & (NEXT_Cann==2)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Miss and 1Cann R->L, [_,_,0] -> [+1,+1,1]
next_state([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
<= (NEXT_Miss==Miss+1) & (NEXT_Cann==Cann+1)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Miss R->L, [_,_,0] -> [+1,_,1]
next_state([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
<= (NEXT_Miss==Miss+1) & (NEXT_Cann==Cann)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Cann R->L, [_,_,0] -> [_,+1,1]
next_state([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
<= (NEXT_Miss==Miss) & (NEXT_Cann==Cann+1)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

####################
# is state legal
is_ok(Miss,Cann)\
<=(Miss >= 0) & (Miss <= 2) & (Cann >= 0) & (Cann <= 2)\
  & is_side_ok(Miss,Cann) & is_side_ok(2-Miss, 2-Cann)
is_side_ok(Miss,Cann) <= (Miss >= Cann)
is_side_ok(Miss,Cann) <= (Miss == 0)

# recursive
from_to([Miss,Cann,Boat], Y, X)\
<= next_state([Miss,Cann,Boat], [NEXT_Miss,NEXT_Cann,NEXT_Boat])\
& (P==is_in([NEXT_Miss,NEXT_Cann,NEXT_Boat], Y)) & (P==True)\
& from_to([NEXT_Miss,NEXT_Cann,NEXT_Boat], Y+[[NEXT_Miss,NEXT_Cann,NEXT_Boat]], X+[[NEXT_Miss,NEXT_Cann,NEXT_Boat]])

from_to([Miss,Cann,Boat], O, X)\
<= (Miss==0) & (Cann==0) & (Boat==0) & (Q==ppp(X))


(Z == (X==[[2,2,1]]) & (from_to([2,2,1], [[2,2,1]], X)))


