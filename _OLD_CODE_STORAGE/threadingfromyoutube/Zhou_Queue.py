import threading
import time
from queue import Queue

def job(l,q):
    for i in range(len(l)):
        l[i] = l[i]**2
    q.put(l)

def main():
    q = Queue()
    threads = []
    data = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
    for i in range(4):
        t = threading.Thread(target = job, args = (data[i],q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    results = []
    for _ in range(4):
        results.append(q.get())

    print(results)

if __name__ == '__main__':
    main()
