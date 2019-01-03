import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle

since = time.time()

print('Loading data...')
data_path = '../data/'
save_dir = '../fake/'
if 'fake' in save_dir:
    print('-' * 45)
    print()
    print(' !' * 22)
    print()
    print('  this is fake world  ' * 2)
    print()
    print(' !' * 22)
    print()
    print('-' * 45)
df = pd.read_csv(data_path + 'train.csv',
                dtype={'msno': 'category',
                       'source_system_tab': 'category',
                       'source_screen_name': 'category',
                       'source_type': 'category',
                       'target': np.uint8,
                       'song_id': 'category'
                       }
                )


train_size = 0.76
length = len(df)

df1 = df.head(int(length * train_size))
# df_fake_test = df.drop(df_fake_train.index)

save_name = 'train'
vers = ''

print('dtypes of df:')
print('>'*20)
print(df1.dtypes)
print('number of columns:', len(df1.columns))
print('number of data:', len(df1))
print('<'*20)
df1.to_csv(save_dir+save_name+vers+'.csv', index=False)
# pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

df2 = df.drop(df1.index)

save_name = 'test'
vers = ''

print('dtypes of df:')
print('>'*20)
print(df2.dtypes)
print('number of columns:', len(df2.columns))
print('number of data:', len(df2))
print('<'*20)
df2.to_csv(save_dir+save_name+vers+'.csv', index=False)


print('All Done.')
if 'fake' in save_dir:
    print('-' * 45)
    print()
    print(' !' * 22)
    print()
    print('  this is fake world  ' * 2)
    print()
    print(' !' * 22)
    print()
    print('-' * 45)
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))