import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'


df1 = pd.read_csv(data_dir+"songs.csv",
                 dtype={'song_id': 'category',
                        'song_length': np.float64,
                        'genre_ids': 'category',
                        'artist_name': 'category',
                        'composer': 'category',
                        'lyricist': 'category',
                        'language': np.float64,
                        # 'name': 'category',
                        # 'isrc': 'category'
                        }
                 )

df2 = pd.read_csv(data_dir+"song_extra_info.csv",
                 dtype={
                        'name': 'category',
                        'isrc': 'category'
                        }
                 )

df = df1.merge(df2, on='song_id', how='outer')
st1 = set(df1['song_id'])
st2 = set(df2['song_id'])
st = set.intersection(st1, st2)
print('len of st:', len(st))
del df1, df2

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
    print('inspecting:', on)
    # print('>'*20)
    print('any null:', df[on].isnull().values.any())
    print('null number:', df[on].isnull().values.sum())
    print(on, 'dtype:', df[on].dtypes)
    # print('describing', on, ':')
    # print(df[on].describe())
    print('<'*20)
    l = df[on]
    s = set(l)
    print('list len:', len(l))
    print('set len:', len(s))
    print()
print('<'*20)
print('<'*20)
print('<'*20)




print('creating custom song.')
save_name = 'song'
vers = ''
d = df.dtypes.to_dict()
print(d)
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)
pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

print('song + song extra done.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

