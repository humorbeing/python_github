import csv
# filename = 'data.csv'
# with open(filename, mode='w') as file:
#     writer = csv.writer(file)
#
#     writer.writerow(['Programming language', 'Designed by', 'Appeared', 'Extension'])
#     writer.writerow(['Python', 'Guido, van, Rossum', '1991', '.py'])  # testing ","
#     # writer.writerow(['Java', 'James Gosling', '1995', '.java'])
#     # writer.writerow(['C++', 'Bjarne Stroustrup', '1985', '.cpp'])
#
#
# with open(filename, mode='a') as file:
#     writer = csv.writer(file)
# #
#     # writer.writerow(['Programming language', 'Designed by', 'Appeared', 'Extension'])
# #     writer.writerow(['Python', 'Guido, van, Rossum', '1991', '.py'])  # testing ","
#     writer.writerow(['Java', 'James Gosling', '1995', '.java'])
#     writer.writerow(['C++', 'Bjarne Stroustrup', '1985', '.cpp'])
# # import panda as pd  # Error is raised, thank you... otherwise, i will more lost
import pandas as pd
# data = pd.read_csv(filename)
# print(data.head())

from argparse import Namespace
# f_name = 'a.csv'
args = Namespace()
args.model = 'a'
args.batch_size = 200
args.epoch = 200
args.model = 'model_name'
args.mode = 'recon'
args.seed = 6
args.input_zero = False
args.loss_function = 'mse'
args.is_cuda = True
args.work_path = './'

df = pd.DataFrame([vars(args)])
print(df.head())
# df.to_csv('a.csv', mode='w')
#
# aa = pd.read_csv('b.csv')
def appendDFToCSV_void(df, csvFilePath, sep=","):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        df1 = pd.read_csv(csvFilePath, sep=sep)
        n = pd.concat([df, df1], axis=0, ignore_index=True)
        n.to_csv(csvFilePath, mode='w', index=False, sep=sep)
        # raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)

from datetime import datetime
import os

def save_args(args, f_name):
    df = pd.DataFrame([vars(args)])
    appendDFToCSV_void(df, f_name)

def log(args, csv_name='log.csv'):
    if not os.path.exists(args.work_path + 'logs'):
        os.makedirs(args.work_path + 'logs')
    t = datetime.now()
    surfix = t.strftime('%Y%m%d-%H%M%S-')
    args.log_time = t.strftime('%Y%m%d%H%M%S')
    name = args.model+'-'+args.mode
    args.log_path = args.work_path+'logs/' + surfix + name + '.txt'
    save_args(args, args.work_path+'logs/'+csv_name)

log(args)
print(args.log_path)