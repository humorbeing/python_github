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
train_size = 0.76
length = len(df)

df = df.head(length * train_size)
# df_fake_test = df.drop(df_fake_train.index)

save_name = 'fake_train'
vers = ''

print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)
# pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

df = df.drop(df.index)

save_name = 'fake_test'
vers = ''

print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)


print('All Done.')
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))