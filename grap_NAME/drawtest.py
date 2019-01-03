import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


with open('a.a','rb') as ff:
    b = pickle.load(ff)

t = len(b)
n = 200
for i in range(t):
    b[i][0] = b[i][0] - 0.01
    b[i][0] = float("{0:.2f}".format(b[i][0]))
    b[i][1] = b[i][1]/n

#print (b)
'''
fig = plt.figure()
axes = plt.gca()
axes.set_ylim([0,1])
ax = fig.add_subplot(111)
line3, = ax.plot(0.5,b,'r-')
line3.set_ydata(b)
plt.show()
'''
x=[]
y=[]
for i in range(t):
    x.append(b[i][0])
    y.append(b[i][1])

plt.plot(x,y)
#plt.axis([0, 0.5, 0, 1])
plt.show()
