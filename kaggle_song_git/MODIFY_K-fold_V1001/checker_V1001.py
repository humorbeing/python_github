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
load_name = 'TEST_dart_goss_rf_gbdt_.csv'
# load_name = 'test_set.csv'
load_name = load_name[:-4]
# print(load_name)
read_from = '../fake/saves/feature/'
dt = pickle.load(open(read_from+load_name+'_dict.save', "rb"))
df = pd.read_csv(read_from+load_name+".csv",
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


# df1 = df[df.index % 2 == 0]
# df2 = df[df.index % 2 == 1]
# # df3 = df[df.index % 3 == 2]
#
# # df1 = df.iloc[0::3, :]
# # df2 = df.iloc[1::3, :]
# # df3 = df.iloc[2::3, :]
#
# print('number of rows:', len(df1))
# df1 = df1[df1['target'] == 1]
# print('number of rows:', len(df1))
# print('number of rows:', len(df2))
# df2 = df2[df2['target'] == 1]
# print('number of rows:', len(df2))
# print('number of rows:', len(df3))
# df3 = df3[df3['target'] == 1]
# print('number of rows:', len(df3))

# K = 2
# dfs = []
# for i in range(K):
#     dfs.append(df[df.index % K == i])
# del df
#
# for i in dfs:
#     print('number of rows:', len(i))
#     i = i[i['target'] == 1]
#     print('number of 1 rows:', len(i))
#     print()


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


