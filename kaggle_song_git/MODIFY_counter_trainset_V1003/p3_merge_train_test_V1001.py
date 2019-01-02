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
save_dir = '../saves/'
df = pd.read_csv(data_path + 'train.csv',
                    dtype={'msno': 'category',
                           'source_system_tab': 'category',
                           'source_screen_name': 'category',
                           'source_type': 'category',
                           'target': np.uint8,
                           'song_id': 'category'
                           }
                    )
dt = pickle.load(open(save_dir+"custom_members_fixed_dict.save", "rb"))
members = pd.read_csv(save_dir+"custom_members_fixed.csv", dtype=dt)
del dt

print('Done loading...')


print('Data merging...')


df = df.merge(members, on='msno', how='left')
del members
load_name = 'custom_song_fixed'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
songs = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
df = df.merge(songs, on='song_id', how='left')
del songs

print('creating train merge.')
save_name = 'train_'
vers = 'merge'
d = df.dtypes.to_dict()
# print(d)
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)
pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

print('train merge done.')

# print('Done merging...')
print('Loading data...')
df = pd.read_csv(data_path + 'test.csv',
                    dtype={'msno': 'category',
                           'source_system_tab': 'category',
                           'source_screen_name': 'category',
                           'source_type': 'category',
                           # 'target': np.uint8,
                           'song_id': 'category',
                           'id': 'category'
                           }
                 )
dt = pickle.load(open(save_dir+"custom_members_fixed_dict.save", "rb"))
members = pd.read_csv(save_dir+"custom_members_fixed.csv", dtype=dt)
del dt

print('Done loading...')


print('Data merging...')


df = df.merge(members, on='msno', how='left')
del members
load_name = 'custom_song_fixed'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
songs = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
df = df.merge(songs, on='song_id', how='left')
del songs

print('creating test merge.')
save_name = 'test_'
vers = 'merge'
d = df.dtypes.to_dict()
# print(d)
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)
pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

# print('test merge done.')
print('p3 merge train, test Done.')
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))