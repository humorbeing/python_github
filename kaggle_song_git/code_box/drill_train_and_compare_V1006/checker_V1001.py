import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'

# load_name = 'train_set.csv'
load_name = 'test_set.csv'
load_name = load_name[:-4]
# print(load_name)
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv",
                 dtype=dt)
del dt


def insert_this(on):
    global df
    on = on[:-4]
    df1 = pd.read_csv('../saves/feature/'+on+'.csv')
    df1.drop('id', axis=1, inplace=True)
    on = on[-10:]
    # print(on)
    df1.rename(columns={'target': on}, inplace=True)
    # print(df1.head(10))
    df = df.join(df1)
    del df1


insert_this('[0.67982]_0.6788_Light_gbdt_1512750240.csv')
insert_this('[0.62259]_0.6246_Light_gbdt_1512859793.csv')

# print(df.head(10))
print()
print('>'*20)
print('>'*20)
print('dtypes of df:')

print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))
# print('<'*20)


# for on in df.columns:
#     print()
#     print('inspecting:', on)
#     # print('>'*20)
#     print('any null:', df[on].isnull().values.any())
#     print('null number:', df[on].isnull().values.sum())
#     print()
#     print(on, 'dtype:', df[on].dtypes)
#     print('describing', on, ':')
#     print(df[on].describe())
#     print('<'*20)
#     l = df[on]
#     s = set(l)
#     print('list len:', len(l))
#     print('set len:', len(s))
#     print()
print('<'*20)
print('<'*20)
print('<'*20)



print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


