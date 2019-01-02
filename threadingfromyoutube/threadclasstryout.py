import threading
import time

class Td(threading.Thread):


    def run(self):
        print('hihihi')

t = Td()
t.start()
