class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        p = Point(self.x + other.x, self.y + other.y)
        return p
    def printMe(self):
        print('(',self.x,',',self.y,')')
    def __str__(self):
        return '( '+str(self.x)+' , '+str(self.y)+' )'
class Rectangle:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    def area(self):
        return abs((self.p1.x-self.p2.x)*(self.p1.y-self.p2.y))
    def plusArea(self,S):
        return self.area()+S.area()
    def addArea(self,S):
        print('Rectangle:[',self,'] + [',S,'] is:')
        isitoverlap = False
        s = self
        b = S
        if self.area() > S.area():
            s, b = b, s
        small = []
        small.append(Point(min(s.p1.x,s.p2.x),min(s.p1.y,s.p2.y)))
        small.append(Point(min(s.p1.x,s.p2.x),max(s.p1.y,s.p2.y)))
        small.append(Point(max(s.p1.x,s.p2.x),max(s.p1.y,s.p2.y)))
        small.append(Point(max(s.p1.x,s.p2.x),min(s.p1.y,s.p2.y)))
        big = []
        big.append(Point(max(b.p1.x,b.p2.x),max(b.p1.y,b.p2.y))) #0
        big.append(Point(max(b.p1.x,b.p2.x),min(b.p1.y,b.p2.y))) #1
        big.append(Point(min(b.p1.x,b.p2.x),min(b.p1.y,b.p2.y))) #2
        big.append(Point(min(b.p1.x,b.p2.x),max(b.p1.y,b.p2.y))) #3



        '''
        for i in small:
            print(i)
        for i in big:
            print(i)
        '''
        for i in range(4):
            if self.pointinarea(Rectangle(big[2],big[0]),small[i]):
                isitoverlap = True
                break
        if not isitoverlap:
            #print('its two Rectangles')
            return self.plusArea(S)
        else:
            checkall = []
            for i in small:
                if self.pointinarea(Rectangle(big[2],big[0]),i):
                    checkall.append(1)
                else:
                    checkall.append(0)
                #checkall.append(self.pointinarea(Rectangle(big[2],big[0]),i))
            if sum(checkall) == 4:
                #print('small one inside big one')
                return Rectangle(big[2],big[0]).area()
            else:
                #print('part overlap')
                if sum(checkall) == 1:
                    #print('one point inside')
                    for i in range(4):
                        if self.pointinarea(Rectangle(big[2],big[0]),small[i]):
                            #print(i)
                            return self.plusArea(S) - Rectangle(small[i],big[i]).area()
                            break
                else:
                    #print('two point inside')
                    gotfirstone = False
                    for i in range(4):
                        if self.pointinarea(Rectangle(big[2],big[0]),small[i]):
                            if gotfirstone:
                                return self.plusArea(S) - Rectangle(small[i],big[i]).area()
                                break
                            else:
                                gotfirstone = True
                                if self.pointinareanotlap(Rectangle(big[2],big[0]),small[i]):
                                    return self.plusArea(S) - Rectangle(small[i],big[i]).area()
                                    break
    def pointinarea(self, Rec, P):
        if Rec.p1.x <= P.x <= Rec.p2.x:
            if Rec.p1.y <= P.y <= Rec.p2.y:
                return True
            else:
                return False
        else:
            return False
    def pointinareanotlap(self, Rec, P):
        if Rec.p1.x < P.x < Rec.p2.x:
            if Rec.p1.y < P.y < Rec.p2.y:
                return True
            else:
                return False
        else:
            return False
    def __str__(self):
        return '( '+str(self.p1.x)+' , '+str(self.p1.y)+' ) , ( '+str(self.p2.x)+' , '+str(self.p2.y)+' )'
def main():
    p = Point(1,5)
    p.printMe()
    print("p = ",p)
    print(p+p)
    q = Point(5,1)
    rec1 = Rectangle(Point(-1,3),Point(3,-2))
    rec2 = Rectangle(Point(4,2),Point(1,-2))
    rec3 = Rectangle(Point(0,3),Point(3,0))
    rec4 = Rectangle(Point(5,3),Point(2,0))
    rec5 = Rectangle(Point(-1,-1),Point(0,0))
    print(rec1)
    print(rec1.area())
    print(rec2)
    print(rec2.area())
    print(rec1.addArea(rec2))
    print(rec3.addArea(rec4))
    print(rec1.addArea(rec1))
    print(rec5.addArea(rec3))
    print(rec5.addArea(rec4))
    #rec1.addArea(rec2)
    #rec3.addArea(rec4)
if __name__ == '__main__':
    main()
