import threading
from queue import Queue
import time

lock = threading.Lock()
i = 0
def p1():
    global i
    for _ in range(10):

        with lock:
            i = 1
            time.sleep(1)
        print('ooooo',i)

def p2():
    global i
    for _ in range(10):
        with lock:
            i = 2
            time.sleep(2)
        print('ttttt', i)

def p3():
    global i
    while True:
        time.sleep(0.5)
        # i = 3
        # with lock:
        #     i = 3
        print(i)

t1 = threading.Thread(target=p1)
t2 = threading.Thread(target=p2)
t3 = threading.Thread(target=p3)
t1.daemon = True
t2.daemon = True
t3.daemon = True
t1.start()
t2.start()
t3.start()
t1.join()
print('*'*20)
t2.join()
print('end')