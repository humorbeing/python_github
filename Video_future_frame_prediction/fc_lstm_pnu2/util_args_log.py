from datetime import datetime
import os
import pandas as pd


def combine_two(from1,from2,comb_to, sep=","):
    df1 = pd.read_csv(from1, sep=sep)
    df2 = pd.read_csv(from2, sep=sep)
    n = pd.concat([df1, df2], axis=0, ignore_index=True)
    n.to_csv(comb_to, mode='w', index=False, sep=sep)

def appendDFToCSV_void(df, csvFilePath, sep=","):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        df1 = pd.read_csv(csvFilePath, sep=sep)
        n = pd.concat([df, df1], axis=0, ignore_index=True, sort=True)
        n.to_csv(csvFilePath, mode='w', index=False, sep=sep)
        # raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        df1 = pd.read_csv(csvFilePath, sep=sep)
        n = pd.concat([df, df1], axis=0, ignore_index=True, sort=True)
        n.to_csv(csvFilePath, mode='w', index=False, sep=sep)
        # raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


def save_args(args, f_name):
    df = pd.DataFrame([vars(args)])
    appendDFToCSV_void(df, f_name)


class Log():
    def __init__(self, args, csv_name='log.csv'):
        if not os.path.exists(args.work_path + 'logs'):
            os.makedirs(args.work_path + 'logs')
        t = datetime.now()
        surfix = t.strftime('%Y%m%d-%H%M%S')
        args.log_time = surfix
        args.this_name = args.model + '-' + args.mode
        args.model_save_file = args.save_path + args.this_name + '.save'
        args.log_path = args.work_path + 'logs/' + surfix + '-' + args.this_name + '.txt'
        save_args(args, args.work_path + 'logs/' + csv_name)
        self.log_file = args.log_path
        with open(self.log_file, 'w'):
            print('opening log file:', self.log_file)

    def log(self, log_string):
        print(log_string)
        with open(self.log_file, 'a') as f:
            f.write(log_string + '\n')

    def end(self):
        print('log is saved in: {}'.format(self.log_file))


if __name__ == '__main__':
    # from argparse import Namespace
    #
    # args = Namespace()
    # args.model = 'a'
    # args.batch_size = 200
    # args.epoch = 200
    # args.model = 'model_name'
    # args.mode = 'recon'
    # args.seed = 6
    # args.input_zero = False
    # args.loss_function = 'mse'
    # args.is_cuda = True
    # args.work_path = './'
    #
    # log = Log(args)
    #
    # s1 = 'Epoch: 0, train loss: 0.038667300964395204, eval loss: 0.037765941893061004'
    # s2 = 'Epoch: 1, train loss: 0.03659273808201154, eval loss: 0.035389104237159096'
    # s3 = 'Epoch: 2, train loss: 0.034583510582645735, eval loss: 0.03329953116675218'
    #
    # log.log(s1)
    # log.log(s2)
    # log.log(s3)
    #
    # log.end()
    p1 = '/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/__SSSSSSandbox/playground_something/a.csv'
    p2 = '/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/__SSSSSSandbox/playground_something/logs/log.csv'
    p3 = '/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/__SSSSSSandbox/playground_something/logs/log1.csv'
    combine_two(p1,p2,p3)

    pass