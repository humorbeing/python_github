from pyDatalog_playground import pyDatalog
pyDatalog.create_terms('X,Y,Z')
pyDatalog.create_terms('Miss,Cann,NEXT_Miss,NEXT_Cann')
pyDatalog.create_terms('Boat,NEXT_Boat')
pyDatalog.create_terms('is_ok,is_side_ok,next_stage,from_to')

#2Miss L->R, [2,_,1] -> [0,_,0]
next_stage([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (Miss==2) & (NEXT_Miss==0) & (NEXT_Cann==Cann)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#2Cann L->R, [_,2,1] -> [_,0,0]
next_stage([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (Cann==2) & (NEXT_Miss==Miss) & (NEXT_Cann==0)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Miss and 1Cann L->R, [_,_,1] -> [-1,-1,0]
next_stage([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (NEXT_Miss==Miss-1) & (NEXT_Cann==Cann-1)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Miss L->R, [_,_,1] -> [-1,_,0]
next_stage([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (NEXT_Miss==Miss-1) & (NEXT_Cann==Cann)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Cann L->R, [_,_,1] -> [_,-1,0]
next_stage([Miss,Cann,1],[NEXT_Miss,NEXT_Cann,0])\
<= (NEXT_Miss==Miss) & (NEXT_Cann==Cann-1)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

####################---------######################

#2Miss R->L, [0,_,0] -> [2,_,1]
next_stage([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
<= (Miss==0) & (NEXT_Miss==2) & (NEXT_Cann==Cann)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#2Cann R->L, [_,0,0] -> [_,2,1]
next_stage([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
<= (Cann==0) & (NEXT_Miss==Miss) & (NEXT_Cann==2)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Miss and 1Cann R->L, [_,_,0] -> [+1,+1,1]
next_stage([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
<= (NEXT_Miss==Miss+1) & (NEXT_Cann==Cann+1)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Miss R->L, [_,_,0] -> [+1,_,1]
next_stage([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
<= (NEXT_Miss==Miss+1) & (NEXT_Cann==Cann)\
  & is_ok(Miss,Cann) & is_ok(NEXT_Miss,NEXT_Cann)

#1Cann R->L, [_,_,0] -> [_,+1,1]
next_stage([Miss,Cann,0],[NEXT_Miss,NEXT_Cann,1])\
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
from_to([Miss,Cann,Boat])\
<= next_stage([Miss,Cann,Boat],
              [NEXT_Miss,NEXT_Cann,NEXT_Boat])\
   & from_to([NEXT_Miss,NEXT_Cann,NEXT_Boat])
from_to([Miss,Cann,Boat])\
<= (Miss==0) & (Cann==0) & (Boat==0)

print(from_to([2,2,1]))