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
    for i in vars(args):
        print('ARGS >>> ' + i + ' :{}'.format(vars(args)[i]))
    df = pd.DataFrame([vars(args)])
    appendDFToCSV_void(df, f_name)

# need a better naming policy
# def naming(args):
#     return 'name'

class Log():
    def __init__(self, args, csv_name='log.csv'):
        if not os.path.exists(args.log_path + 'logs'):
            os.makedirs(args.log_path + 'logs')
        t = datetime.now()
        surfix = t.strftime('%Y%m%d-%H%M%S')
        args.log_time = surfix
        args.this_name = surfix + '-' + naming(args)
        args.model_save_file = args.save_path + args.this_name + '.save'
        args.log_file = args.log_path + 'logs/' + args.this_name + '.txt'
        save_args(args, args.log_path + 'logs/' + csv_name)
        self.log_file = args.log_file
        with open(self.log_file, 'w'):
            print('opening log file:', self.log_file)

    def log(self, log_string):
        print(log_string)
        with open(self.log_file, 'a') as f:
            f.write(log_string + '\n')

    def end(self):
        print('log is saved in: {}'.format(self.log_file))


def naming(args):
    name = ''
    if args.optimizer == 'rmsprop':
        name += 'O_RMS-'
    else:
        name += 'O_Adm-'
    # if args.model == 'model_name':
    #     name += 'M1_'
    name += 'M1.1-'

    # args.last_activation = 'non'  # '100s' / 'sigmoid' / 'non'
    # args.loss_function = 'mse'  # 'mse' / 'bce'
    if args.last_activation == 'non':
        name += 'N'
    elif args.last_activation == 'sigmoid':
        name += 'S'
    else:
        name += '100'
    if (not (args.last_activation != 'non')) and (args.loss_function == 'bce'):
        name += 'B'
    else:
        name += 'M'
    name += '-'
    if args.zero_input:
        name += 'Zt'
    else:
        name += 'Zf'
    name += '-'
    if args.mode == 'recon':
        name += 'R'
    elif args.mode == 'pred':
        name += 'P'
    else:
        name += 'B'
    return name

if __name__ == '__main__':
    # from argparse import Namespace
    #
    #
    # # ///////////////////////////////////////////////////////
    #
    # args = get_args()
    #
    # log = Log(args, 'test.csv')
    #
    # s1 = 'Epoch: 0, train loss: 0.038667300964395204, eval loss: 0.037765941893061004'
    # s2 = 'Epoch: 1, train loss: 0.03659273808201154, eval loss: 0.035389104237159096'
    # s3 = 'Epoch: 2, train loss: 0.034583510582645735, eval loss: 0.03329953116675218'
    #
    # log.log(s1)
    # log.log(s2)
    # log.log(s3)
    # #
    # log.end()
    # p3 = '/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/Video_future_frame_prediction/fc_lstm_n202_ubu/gather_logs_here/all_logs/fc_lstm_logs.csv'
    # p1 = p3
    # p2 = '/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/Video_future_frame_prediction/fc_lstm_n202_ubu/gather_logs_here/recieved_logs/logs_colab_r/log.csv'
    # combine_two(p1,p2,p3)
    pass