import threading


def do_this():
    global x


    while True:
        pass

def do_after():
    global x


    while x < 600:
        x += 1
    print (x)

def main():

    global x
    x = 0
    main_thread = threading.enumerate()[0]

    our_thread = threading.Thread(target = do_this, name = 'Our Thread')
    our_thread.daemon = True

    our_thread.start()

    print (our_thread.isDaemon())




if __name__ == '__main__':
    main()
