import threading

def do_this():
    print ('This is our thread!')

def main():

    our_thread = threading.Thread(target = do_this)
    our_thread.start()

    print(threading.active_count())
    print(threading.enumerate())
    print (threading.current_thread())
if __name__ == '__main__':
    main()
