#       Cab-Wolf-Goat-Man
start = [1,1,1,1]
goal = [0,0,0,0]

rule = [
        [0,2],
        [1,2]
        ]
status = start
tracker = []
walk = []
deadend = []
gameover = False
count = 0
def goEast():
    global count
    count += 1
    if count > 50:
        gameover = True
    walk.append(status)
    if not gameover:
        tracker.append(status)
        bringWith()
        goWest()
    else:
        print('  -( T - T )-  '*4)
        print(walk)
        print('  -( T - T )-  '*4)
def goWest():
    walk.append(status)
    if not gameover:
        tracker.append(status)
        bringWith()
        if status == goal:
            print('we made it, shortest path is')
            tracker.append(status)
            print(tracker)
        else:
            goEast()
    else:
        print('  -( T - T )-  '*4)
        print(walk)
        print('  -( T - T )-  '*4)
def bringWith():
    candidate = []
    for i in range(3):
        if status[3] == status[i]:
            candidate.append(i)
    candidate.append(3)
    if options(candidate):
        if candidate[0] == 3:
            sta[3] = zerotoone(sta[3])
        else:
            sta[3] = zerotoone(sta[3])
            sta[candidate[0]] = zerotoone(sta[candidate[0]])
    else:
        backtrack()

def backtrack():
    global status
    print('in backtrack')
    print('check if {} is in deadend'.format(status))
    print(deadend)
    print('------------deadend------------')
    if not status in deadend:
        deadend.append(status)
    print('checking tracker  ---------before')
    print(tracker)
    if tracker:
        del tracker[-1]
    if tracker:
        status = tracker[-1]
    else:
        print('*/'*20)
        print('  Game over!  '*3)
        print('*/'*20)
        gameover = True
    print('tracker after change')
    print(tracker)

def options(S):
    sta = status
    viableoption = []
    for i in S:
        if i == 3:
            sta[3] = zerotoone(sta[3])
        else:
            sta[3] = zerotoone(sta[3])
            sta[i] = zerotoone(sta[i])
        print('check if {} is in deadend'.format(sta))
        print(deadend)
        print('------------deadend------------')
        if not sta in deadend:
            if isitsafe(sta):
                print('we got a viableoption {}'.format(i))
                viableoption.append(i)
            else:
                deadend.append(sta)
    if viableoption:
        print('the options are as follows')
        print(viableoption)
        return viableoption
    else:
        print('bad end, returning False instead of options')
        return False

def isitsafe(s):
    alone = []
    safe = True
    dangerous = True
    for i in range(3):
        if s[3] != s[i]:
            alone.append(i)
    if alone:
        t = []
        print(alone)
        for i in alone:

            if i == 0:
                t.append('Cabb')
            elif i == 1:
                t.append('Wolf')
            elif i == 2:
                t.append('Goat')
            else:
                t.append('Monstor')

        print('All alone: ', end='')
        print(t)
        for i in range(len(rule)):
            dangerous = True
            for j in rule[i]:
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

def zerotoone(s):
    if s == 0:
        return 1
    else:
        return 0

goWest()
