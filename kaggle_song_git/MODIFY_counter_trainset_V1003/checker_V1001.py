import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'

load_name = 'custom_members_fixed.csv'
load_name = 'custom_song_fixed.csv'
load_name = 'train_set.csv'
# load_name = 'test_set.csv'
load_name = load_name[:-4]
# print(load_name)
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv",
                 dtype=dt)
del dt


print()
print('>'*20)
print('>'*20)
print('dtypes of df:')

print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))
# print('<'*20)


for on in df.columns:
    print()
    print('inspecting:'.ljust(20), on)
    # print('>'*20)
    print('any null:'.ljust(15), df[on].isnull().values.any())
    print('null number:'.ljust(15), df[on].isnull().values.sum())
    print(on, 'dtype:', df[on].dtypes)
    print('describing', on, ':')
    print(df[on].describe())
    print('-'*20)
    l = df[on]
    s = set(l)
    print('list len:'.ljust(20), len(l))
    print('set len:'.ljust(20), len(s))
    print()
print('<'*20)
print('<'*20)
print('<'*20)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


