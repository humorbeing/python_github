import threading
import time


def job():
    for i in range(5):
        time.sleep(1)
        breed()
def breed():
    if threading.active_count()<50:
        #print('pop is {}. here is a baby'.format(threading.active_count()))
        time.sleep(1)

        t = threading.Thread(target = job)
        t.daemon = True
        t.start()

    else:
        pass
        #print('too crowded')

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



if __name__ == "__main__":
    main()
