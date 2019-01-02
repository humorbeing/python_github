import threading


def do_this():
    global dead
    x = 0
    print ('This is our thread!')
    while not dead:
        x += 1
        pass
    print (x)

def main():

    global dead
    dead = False

    our_thread = threading.Thread(target = do_this, name = 'Our Thread')
    #our_thread.daemon = True
    our_thread.start()

    print(threading.active_count())
    print(threading.enumerate())

    print(our_thread.is_alive())

    input("Hit enter to die") #use cmd line here.
    dead = True

    print(our_thread.is_alive())

    input('Need a 2nd enter')

    print(our_thread.is_alive())

if __name__ == '__main__':
    main()
