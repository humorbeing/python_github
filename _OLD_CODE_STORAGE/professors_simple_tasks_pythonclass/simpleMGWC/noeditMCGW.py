    #       Cab-Wolf-Goat-Man
class Game:
    def __init__(self):
        self.start = [1,1,1,1]
        self.goal = [0,0,0,0]

        self.rule = [
                     [0,2],
                     [1,2]
                    ]
        self.operators()
        #self.deadend = []


    def operators(self):
        self.status = self.givelist(self.start)
        self.tracker = []
        self.walk = []
        self.deadend = []
        self.gameover = False
        self.count = 0
        self.tem = []
        self.sta = []

    def run(self):
        if self.status[3] == 1:
            self.goWest()
        else:
            self.goEast()

    def goEast(self):
        print("<<<<<<<<<<<<----------------")
        print('I am at West, moving to East')
        print(self.status)
        #global count
        self.count += 1
        print('counter is {}'.format([self.count]*20))
        print('counter is {}'.format([self.count]*20))
        if self.count > 6:
            print('counter is {}'.format([self.count]*20))
            print('counter is {}'.format([self.count]*20))
            self.gameover = True
        self.update(self.walk,self.status)
        if not self.gameover:
            self.updatetracker(self.status)
            print('OOOOOOOO tracking yo!')
            print(self.tracker)
            print('OOOOOOOOOOOOOOOOOOOOOOOOO')
            self.bringWith()
            self.goWest()
        else:
            print('  -( T - T )-  '*4)
            print(self.walk)
            print('  -( T - T )-  '*4)

    def goWest(self):
        print("---------------->>>>>>>>>>>>>")
        print('I am at East, moving to West')
        print(self.status)
        self.update(self.walk,self.status)
        if not self.gameover:
            self.updatetracker(self.status)
            print('OOOOOOOO tracking yo!')
            print(self.tracker)
            print('OOOOOOOOOOOOOOOOOOOOOOOOO')
            self.bringWith()
            if self.status == self.goal:
                print('we made it, shortest path is')
                self.updatetracker(self.status)
                print(self.tracker)
            else:
                self.goEast()
        else:
            print('  -( T - T )-  '*4)
            print(self.walk)
            print('  -( T - T )-  '*4)

    def bringWith(self):
        print('--let me see what can I carry!')
        candidate = []
        for i in range(3):
            if self.status[3] == self.status[i]:
                candidate.append(i)
        candidate.append(3)
        print('--They are: ',end=' ')
        print(candidate)
        print('--Which one is safe to bring along?')
        viable = self.options(candidate)
        if viable:
            print('---let me try {}'.format(viable[0]))
            if viable[0] == 3:
                self.status[3] = self.zerotoone(self.status[3])
            else:
                self.status[3] = self.zerotoone(self.status[3])
                self.status[viable[0]] = self.zerotoone(self.status[viable[0]])
        else:
            print('---NO!,no options here. T - T')
            self.backtrack()

    def options(self, S):
        print('----Hmm, I need to think, give me a second(in options funcion)')
        sta = self.givelist(self.status)
        #print(self.status)
        #print(sta)
        viableoption = []
        for i in S:
            print('----checking item {}'.format(i))
            if i == 3:
                sta[3] = self.zerotoone(sta[3])
            else:
                sta[3] = self.zerotoone(sta[3])
                sta[i] = self.zerotoone(sta[i])

            print('----check if {} is in deadend'.format(sta))
            print(self.deadend)
            print('------------deadend------------')
            if not sta in self.deadend:
                print('----looking good, have we done it before?')
                if not sta in self.tracker:
                    print('----seems good.....is it safe?')
                    if self.isitsafe(sta):
                        print('----we got a viable option {}, thank god'.format(i))
                        viableoption.append(i)

                    else:
                        print('----!!!jeez someone would have died there. let me write it down')
                        print(self.deadend)
                        print('------------before update------------')
                        #self.tem = sta
                        #self.deadend.append(sta)
                        self.updateexp(sta)
                        print(self.deadend)
                        print('------------after update------------')

            #print('----reseting: ',end=' ')
            #print(sta)
            #print('----with: ',end=' ')
            #print(self.status)
            #print(self.start)
            sta = self.givelist(self.status)
            #print('----reseted: ',end=' ')
            #print(sta)

        if viableoption:
            print('----so the options are as follows')
            print(viableoption)
            return viableoption
        else:
            print('----hmm,bad end, returning False instead of options')
            return False

    def backtrack(self):
        print('-----in backtrack')
        print('-----check if {} is in deadend'.format(self.status))
        print(self.deadend)
        print('-----I think {} is the answer.'.format(self.status in self.deadend))
        print('------------deadend------------')
        if not self.givelist(self.status) in self.deadend:
            print('-------before adding--------')
            print(self.deadend)
            print('-------before adding--------')
            self.update(self.deadend,self.status)
            print('-------after adding--------')
            print(self.deadend)
            print('-------after adding--------')
        print('-----checking tracker  ---------before')
        print(self.tracker)
        if self.tracker:
            del self.tracker[-1]
            print('-----tracker after change')
            print(self.tracker)
        if self.tracker:
            self.status = self.givelist(self.tracker[-1])
        else:
            print('*/'*20)
            print('  Game over!  '*3)
            print('*/'*20)
            self.gameover = True

    #########functions###########
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

    def isitsafe(self,s):
        alone = []
        safe = True
        dangerous = True
        for i in range(3):
            if s[3] != s[i]:
                alone.append(i)
        if alone:
            t = []
            #print(alone)
            for i in alone:

                if i == 0:
                    t.append('Cabb')
                elif i == 1:
                    t.append('Wolf')
                elif i == 2:
                    t.append('Goat')
                else:
                    t.append('Monstor')

            print('----------> All alone: ', end='')
            print(t)
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
            print('----------I think, it will be fine!')
            return True
        else:
            print('----------Nope.')
            return False

    def zerotoone(self,s):
        if s == 0:
            return 1
        else:
            return 0

    def updateexp(self,s):
        if not s in self.deadend:
            self.update(self.deadend,s)
game = Game()
game.run()
