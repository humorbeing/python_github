from datetime import datetime
import os


class Log():
    def __init__(self, name, path='./'):
        surfix = datetime.now().strftime('%Y%m%d-%H-%M-%S-')
        self.log_file = path+'logs/' + surfix + name + '.txt'
        if not os.path.exists(path+'logs'):
            os.makedirs(path+'logs')
        with open(self.log_file, 'w'):
            print('opening log file:', self.log_file)

    def log(self, log_string):
        print(log_string)
        with open(self.log_file, 'a') as f:
            f.write(log_string + '\n')

    def end(self):
        print('log is saved in: {}'.format(self.log_file))
