import threading


def do_this():
    global x,lock

    with lock:
        while x < 300:
            x += 1
        print (x)

def do_after():
    global x,lock

    with lock:
        while x < 600:
            x += 1
        print (x)

def main():

    global x,lock
    x = 0
    lock = threading.Lock()
    our_thread = threading.Thread(target = do_this, name = 'Our Thread')
    our_thread.start()

    our_next_thread = threading.Thread(target = do_after, name = 'Our Next Thread')
    our_next_thread.start()




if __name__ == '__main__':
    main()
