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


print('Done loading...')


print('Data merging...')

dt = pickle.load(open(save_dir+"custom_members_fixed_dict.save", "rb"))
members = pd.read_csv(save_dir+"custom_members_fixed.csv", dtype=dt)
del dt
df = df.merge(members, on='msno', how='left')
del members

df.drop([
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'msno',
    'song_id',
],
    axis=1, inplace=True)
print('creating train set.')
save_name = 'train_'
vers = 'set'
d = df.dtypes.to_dict()
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)
pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

print('done.')


# print('train merge done.')
#
# print('p3 merge train, test Done.')
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))