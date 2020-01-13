import threading
import time

def breed():
    if threading.active_count()<50:
        time.sleep(1)
        t = threading.Thread(target = breed)
        t.daemon = True
        t.start()
        while True:
            pass
    else:
        pass
def showpop():
    while True:
        time.sleep(1)
        print('pop is {}'.format(threading.active_count()))


def main():
    t = threading.Thread(target = breed)
    t.start()

    tp = threading.Thread(target = showpop)
    tp.daemon = True
    tp.start()
    input()

if __name__ == '__main__':
    main()
