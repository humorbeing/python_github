import matplotlib.pyplot as plt
x25 =  [0,   50,   100,   150,   200,   250,   300,   350,   400,   450, 1000]
y25f = [0, 0.23,  4.18,  11.3,  15.1,   19,   19,   19,   19,   19,  999]
y25e = [0, 0.01,  0.20,  0.56,  0.83,  3.91,  6.92,  10.0,  17.6,   19,  999]
y25d = [0, 0.01,  0.01,  0.03,  0.05,  0.31,  0.51,  0.11,  1.14,  1.67, 14.8]

x50 =  [0,   50,   100,   150,   200,   250,   300,   350,   400,   450, 1000]
y50f = [0, 0.37,  6.07,  31.9,  15.1,   19,   999,   999,   999,   999,  999]
y50e = [0, 0.03,  0.38,  1.41,  5.35,  7.05,  18.3,   19,   999,   999,  999]
y50d = [0, 0.01,  0.03,  0.02,  0.21,  0.07,  0.13,  0.85,  1.23,  1.77, 17.5]

x75 =  [0,   50,   100,   150,   200,   250,   300,   350,   400,   450, 1000]
y75f = [0, 0.57,  16.2,   19,   999,   999,   999,   999,   999,   999,  999]
y75e = [0, 0.06,  0.74,  2.43,  7.41,  15.1,   19,   999,   999,   999,  999]
y75d = [0, 0.01,  0.01,  0.05,  0.07,  0.37,  0.61,  0.26,   0.4,  1.97, 14.8]

plt.plot(x25, y25f, 'g--', lw=5, label='FF:low')
plt.plot(x25, y25e, 'b--', lw=5, label='EK:low')
plt.plot(x25, y25d, 'r--', lw=5, label='DD:low')
plt.plot(x50, y50f, 'g:', lw=5, label='FF:mid')
plt.plot(x50, y50e, 'b:', lw=5, label='EK:mid')
plt.plot(x50, y50d, 'r:', lw=5, label='DD:mid')
plt.plot(x75, y75f, 'g', lw=3, label='FF:high')
plt.plot(x75, y75e, 'b', lw=3, label='EK:high')
plt.plot(x75, y75d, 'r', lw=3, label='DD:high')
plt.legend(loc=0)
plt.grid()
plt.xlabel('Number of Vertices: with low, mid, high density')
plt.ylabel('Time cost for Maxflow (second)')
plt.title('Ford-F(FF), Edmonds-K(EK), Dinic(DD) Algorithm Compare')
plt.xlim(0, 1050)
plt.ylim(0, 15)
plt.show()
