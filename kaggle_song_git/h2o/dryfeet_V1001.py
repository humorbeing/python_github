import pandas as pd
import time
import numpy as np
import pickle
import h2o

since = time.time()
h2o.init(nthreads=-1)

data_dir = '../data/'
save_dir = '../saves/'
read_from = save_dir

load_name = 'custom_members_fixed.csv'
load_name = 'custom_song_fixed.csv'
load_name = 'train_set.csv'
# load_name = 'test_set.csv'
load_name = 'train_best.csv'
load_name = load_name[:-4]

dt = pickle.load(open(read_from+load_name+'_dict.save', "rb"))
df = pd.read_csv(read_from+load_name+".csv", dtype=dt)
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
    # print('describing', on, ':')
    # print(df[on].describe())
    print('-'*20)
    l = df[on]
    s = set(l)
    print('list len:'.ljust(20), len(l))
    print('set len:'.ljust(20), len(s))
    print()
print('<'*20)
print('<'*20)
print('<'*20)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')

print()
print('Our guest selection:')
print(df.dtypes)
print('number of columns:', len(df.columns))
print()

length = len(df)
train_size = 0.76
train_df = df.head(int(length*train_size))
val_df = df.drop(train_df.index)
del df

train_hf = h2o.H2OFrame(train_df)
del train_df
val_hf = h2o.H2OFrame(val_df)
del val_df

features = list(train_hf.columns)
features.remove('target')

print(train_hf.head(5))
print(val_hf.shape)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
