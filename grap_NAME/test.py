import random #todo 30% of x, this comes from pythons own package.
import matplotlib.pyplot as plt #draw graph, if having problem installing this. google it or [geemguang@pusan.ac.kr] e-mail me.
dot = {} #create vertexes(vertices)
global bugged #to check connected
v = 5 #create how many vertexes(vertices)
n = 5#!!!5000!!! takes 20 mins to run///this is run how many times, if n=50000, like in the assignment, it takes too long to execute it. professor told me to lower it to 5000.
p = 0.01 #starting probility
addp = 0.01 #probility accumulating rate
#this is checking 'is this net connected' function.
def bug_it(breedground=0,comingfrom=0):
    global bugged
    if bugged[breedground]:
        pass
    else:
        bugged[breedground]=True
        for target in range(v):
            if target != breedground and target != comingfrom and not bugged[target]:
                if dot[breedground][target]['connected'] and not bugged[target]:
                    bug_it(target,breedground)
#make bonds(edgs) as given probiblity, apply to all vertexes(vertices) ,but preventing trying same vertexes(vertices) twice. -- like (u,x) --[prevent]:trying u -> x and x -> u
def makebonds(p):
    for i in range(v):#dot
        for j in range(v):#target
            if dot[i][j]['tried'] and dot[j][i]['tried']:
                pass
            else:
                dot[i][j]['tried'] = True
                dot[j][i]['tried'] = True
                if i == j:
                    dot[i][j]['connected'] = True
                elif random.random() < p:
                    dot[i][j]['connected'] = True
                    dot[j][i]['connected'] = True
#init new vertexes(vertices), make them ready to get tested(in this program, its get bugged).
def makedots(p):
    for i in range(v):
        dot[i] = {j:{'tried':False,'connected':False} for j in range(v)}
    makebonds(p)
#count how many vertexes(vertices)-nets(s) are connected in all nets(n), return Pr(w(g)=1) = ?? ,which is S/N from assignment.
def s(p):
    global bugged
    score = 0
    for _ in range(n):
        makedots(p)
        bugged = {i:False for i in range(v)}
        bug_it()
        isconnected = True
        for i in range(v):
            if not bugged[i]:
                isconnected = False
                break
        if isconnected:
            score += 1
    return score/n
#this is main function.
if __name__ == '__main__':
    scores = []
    while p<0.51:
        score = s(p)
        scores.append(score)
        p += addp
    plt.plot([(i+1)/100 for i in range(50)],scores,label="tetsetsetset")
    plt.show()
