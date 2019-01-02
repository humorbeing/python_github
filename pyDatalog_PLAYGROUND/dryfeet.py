from pyDatalog_playground import pyDatalog


pyDatalog.create_terms('X,Y,Z,O,LM,LC,RM,RC,Boat,Lboat,Rboat')
pyDatalog.create_terms('LnM,LnC,RnM,RnC')
pyDatalog.create_terms('LfM,LfC,RfM,RfC')
pyDatalog.create_terms('xin,yout,isok,issideok,next_stage,from_to')
# X.in_((0,1,2,3,4))
# print(X.in_((0,1,2,3,4)))
xin['1'] = 5
print(xin[Y]>4)
print(X.in_(range(5)) & (X<2))
# print(X>1)

isok(LM,LC,RM,RC) <= (LM >= 0) & (LC >= 0) & (RM >= 0) & (RC >= 0) & issideok(LM,LC) & issideok(RM, RC)
issideok(X,Y) <= (X >= Y)
issideok(X,Y) <= (X == 0)
print(X.in_(range(3)) & Y.in_(range(3)) & Z.in_(range(2)) & O.in_(range(2)) & isok(X,Y,Z,O))
print(X.in_(range(2)) & Y.in_(range(2)) & issideok(X,Y))

# next_stage(LM,LC,RM,RC,LnM,LnC,RnM,RnC) <= isok(LM,LC,RM,RC) & (LM==LnM) & (RM==RnM) & (LnC==LC-2) & (RnC==RC+2) & isok(LnM,LnC,RnM,RnC)
# # next_stage(LM,LC,RM,RC,LnM,LnC,RnM,RnC) <= isok(LM,LC,RM,RC) & (LM==LnM) & (RM==RnM) & (LnC==LC) & (RnC==RC) & isok(LnM,LnC,RnM,RnC)
# print(next_stage(2,2,0,0,X,Y,Z,O))
# xin(X, Y) <= (X==0) & (Y==1)
# print(xin(0, LnC))

# next_stage([LM,LC,RM,RC],[LnM,LnC,RnM,RnC]) <= isok(LM,LC,RM,RC) & (LM==LnM) & (RM==RnM) & (LnC==LC-2) & (RnC==RC+2) & isok(LnM,LnC,RnM,RnC)
# # next_stage(LM,LC,RM,RC,LnM,LnC,RnM,RnC) <= isok(LM,LC,RM,RC) & (LM==LnM) & (RM==RnM) & (LnC==LC) & (RnC==RC) & isok(LnM,LnC,RnM,RnC)
# print(next_stage([2,2,0,0],[X,Y,Z,O]))

# next_stage([LM,LC,RM,RC,0],[LnM,LnC,RnM,RnC,1]) <= isok(LM,LC,RM,RC) & (LM==LnM) & (RM==RnM) & (LnC==LC-2) & (RnC==RC+2) & isok(LnM,LnC,RnM,RnC)
# # next_stage(LM,LC,RM,RC,LnM,LnC,RnM,RnC) <= isok(LM,LC,RM,RC) & (LM==LnM) & (RM==RnM) & (LnC==LC) & (RnC==RC) & isok(LnM,LnC,RnM,RnC)
# print(next_stage([2,2,0,0,0],[X,Y,Z,O,1]))
#2M L->R
next_stage([LM,LC,RM,RC,1],[LnM,LnC,RnM,RnC,0])\
<= (LnM==LM-2) & (RnM==RM+2) & (LnC==LC) \
   & (RnC==RC) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)
#2C L->R
next_stage([LM,LC,RM,RC,1],[LnM,LnC,RnM,RnC,0]) <= (LnM==LM) & (RnM==RM) & (LnC==LC-2) & (RnC==RC+2) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)
#1M1C L->R
next_stage([LM,LC,RM,RC,1],[LnM,LnC,RnM,RnC,0]) <= (LnM==LM-1) & (RnM==RM+1) & (LnC==LC-1) & (RnC==RC+1) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)
#1M L->R
next_stage([LM,LC,RM,RC,1],[LnM,LnC,RnM,RnC,0]) <= (LnM==LM-1) & (RnM==RM+1) & (LnC==LC) & (RnC==RC) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)
#1C L->R
next_stage([LM,LC,RM,RC,1],[LnM,LnC,RnM,RnC,0]) <= (LnM==LM) & (RnM==RM) & (LnC==LC-1) & (RnC==RC+1) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)
####################################################
#2M R->L
next_stage([LM,LC,RM,RC,0],[LnM,LnC,RnM,RnC,1]) <= (LnM==LM+2) & (RnM==RM-1) & (LnC==LC) & (RnC==RC) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)
#2C R->L
next_stage([LM,LC,RM,RC,0],[LnM,LnC,RnM,RnC,1]) <= (LnM==LM) & (RnM==RM) & (LnC==LC+2) & (RnC==RC-2) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)
#1M1C R->L
next_stage([LM,LC,RM,RC,0],[LnM,LnC,RnM,RnC,1]) <= (LnM==LM+1) & (RnM==RM-1) & (LnC==LC+1) & (RnC==RC-1) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)
#1M R->L
next_stage([LM,LC,RM,RC,0],[LnM,LnC,RnM,RnC,1]) <= (LnM==LM+1) & (RnM==RM-1) & (LnC==LC) & (RnC==RC) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)
#1C R->L
next_stage([LM,LC,RM,RC,0],[LnM,LnC,RnM,RnC,1]) <= (LnM==LM) & (RnM==RM) & (LnC==LC+1) & (RnC==RC-1) & isok(LnM,LnC,RnM,RnC) & isok(LM,LC,RM,RC)


print(next_stage([2,2,0,0,1],[X,Y,Z,O,Boat]))
print(next_stage([2,2,0,0,1],[2,1,0,1,0]))
print(next_stage([1,0,0,0,1],[0,0,1,0,0]))

from_to([LM,LC,RM,RC,Lboat],
        [LfM,LfC,RfM,RfC,Boat]) <= next_stage([LM,LC,RM,RC,Lboat],[LnM,LnC,RnM,RnC,Rboat]) & from_to([LnM,LnC,RnM,RnC,Rboat],[LfM,LfC,RfM,RfC,Boat])
from_to([LM,LC,RM,RC,Lboat],[LfM,LfC,RfM,RfC,Boat]) <= (LM==LfM) & (LC==LfC) & (RM==RfM) & (RC==RfC) & (Lboat==Boat)
# (from_to([LM,LC,RM,RC,Lboat],[LM,LC,RM,RC,Lboat]))
# from_to([LM,LC,RM,RC,Lboat],[LfM,LfC,RfM,RfC,Boat],X) <= next_stage([LM,LC,RM,RC,Lboat],[LnM,LnC,RnM,RnC,Rboat]) & from_to([LnM,LnC,RnM,RnC,Rboat],[LfM,LfC,RfM,RfC,Boat],[[LM,LC,RM,RC,Lboat],[LnM,LnC,RnM,RnC,Rboat]])
# from_to([LM,LC,RM,RC,Lboat],[LfM,LfC,RfM,RfC,Boat],X) <= (LM==LfM) & (LC==LfC) & (RM==RfM) & (RC==RfC) & (Lboat==Boat)

print(from_to([2,2,0,0,1],[2,1,0,1,0]))
print(from_to([2,2,0,0,1],[2,1,0,1,1]))
# print(from_to([2,2,0,0,1],[2,2,0,1,1]))


# print(from_to([2,2,0,0,1],[2,1,0,1,0]).data[0])
# print(from_to([1,0,0,0,1],[0,0,1,0,0]))
# print(from_to([1,0,0,0,1],[1,0,0,0,1]))

# from_to(X,X) <= (X==1)
# print(from_to(1,1))

# print(from_to([2,2,0,0,1],[0,0,2,2,0], X))

# print(from_to([2,2,0,0,1],[2,1,0,1,0], X))