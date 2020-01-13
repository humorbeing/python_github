    #       Cab-Wolf-Goat-Man
class Game:
    def __init__(self):
        self.start = [1,1,1,1]
        self.goal = [0,0,0,0]
        self.rule = [
                     [0,2],
                     [1,2]
                    ]
        self.status = self.givelist(self.start)
        self.tracker = []
        self.deadend = []

    def run(self):
        self.updatetracker(self.status)
        self.bringWith()
        if self.status == self.goal:
            self.updatetracker(self.status)
            self.showtracker()
        else:
            self.run()

    def bringWith(self):
        candidate = []
        for i in range(3):
            if self.status[3] == self.status[i]:
                candidate.append(i)
        candidate.append(3)
        viable = self.options(candidate)
        if viable:
            if viable[0] == 3:
                self.status[3] = self.zerotoone(self.status[3])
            else:
                self.status[3] = self.zerotoone(self.status[3])
                self.status[viable[0]] = self.zerotoone(self.status[viable[0]])
        else:
            self.backtrack()

    def options(self, S):
        sta = self.givelist(self.status)
        viableoption = []
        for i in S:
            if i == 3:
                sta[3] = self.zerotoone(sta[3])
            else:
                sta[3] = self.zerotoone(sta[3])
                sta[i] = self.zerotoone(sta[i])
            if not sta in self.deadend:
                if not sta in self.tracker:
                    if self.isitsafe(sta):
                        viableoption.append(i)
                    else:
                        self.updateexp(sta)
            sta = self.givelist(self.status)
        if viableoption:
            return viableoption
        else:
            return False

    def backtrack(self):
        if not self.status in self.deadend:
            self.update(self.deadend,self.status)
        del self.tracker[-1]
        self.status = self.givelist(self.tracker[-1])

    #########Necessary Functions###########
    def isitsafe(self,s):
        alone = []
        safe = True
        dangerous = True
        for i in range(3):
            if s[3] != s[i]:
                alone.append(i)
        if alone:
            for i in range(len(self.rule)):
                dangerous = True
                for j in self.rule[i]:
                    if not j in alone:
                        dangerous = False
                        break
                if dangerous:
                    safe = False
                    break
        if safe:
            return True
        else:
            return False
    def zerotoone(self,s):
        if s == 0:
            return 1
        else:
            return 0
    #########list manager##########
    def update(self,l,s):
        b = [0 for i in range(len(s))]
        for i in range(len(s)):
            b[i]=s[i]
        l.append(b)
    def updatetracker(self,s):
        if not s in self.tracker:
            self.update(self.tracker,s)
    def givelist(self, s):
        b = [0 for i in range(len(s))]
        for i in range(len(s)):
            b[i]=s[i]
        return b
    def updateexp(self,s):
        if not s in self.deadend:
            self.update(self.deadend,s)
    ##########to describe##########
    def showtracker(self):
        for i in range(len(self.tracker)-1):
            self.discribmove(self.tracker[i],self.tracker[i+1])
    def discribmove(self,a,b):
        start = " [- -]  "
        if a[3] == 0:
            fromto = "<<---"
        else:
            fromto = "--->>"
        carry=[]
        for i in range(3):
            if a[i] != b[i]:
                carry.append(i)
        if len(carry) == 0:
            carrying = '       '
        elif len(carry) == 1:
            carrying = self.itemname(carry[0]) + " "
        s = start+carrying+fromto
        print()
        print(s)
    def itemname(self, n):
        if n == 0:
            return 'Cabbage'
        elif n == 1:
            return ' Wolf  '
        else:
            return ' Goat  '

game = Game()
game.run()
