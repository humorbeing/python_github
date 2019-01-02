import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


# data_dir = '../data/'
# save_dir = '../saves/'
#
# load_name = 'custom_members_fixed.csv'
# load_name = 'custom_song_fixed.csv'
# load_name = 'train_set.csv'
# load_name = 'test_set.csv'
# load_name = load_name[:-4]
# # print(load_name)
# dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
# df = pd.read_csv(save_dir+load_name+".csv",
#                  dtype=dt)
# del dt
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
df = pd.read_csv(save_dir + 'fake_test.csv',
                dtype={'msno': 'category',
                       'source_system_tab': 'category',
                       'source_screen_name': 'category',
                       'source_type': 'category',
                       'target': np.uint8,
                       'song_id': 'category'
                       }
                )
# df2 = pd.read_csv(save_dir + 'fake_train.csv',
#                 dtype={'msno': 'category',
#                        'source_system_tab': 'category',
#                        'source_screen_name': 'category',
#                        'source_type': 'category',
#                        'target': np.uint8,
#                        'song_id': 'category'
#                        }
#                 )
# df = df_songs
# df = df_custom_song_data
# df = df_songs_extra
# on = 'song_length'
# on = 'source_system_tab'
on = 'msno'
# on = 'source_type'
# on = 'artist_name'
# on = 'composer'
# on = 'composer'
# on = 'language'
# on = 'name'
# on = 'isrc'
# on = 'song_year'
# new_on = 'source_system_tab_guess'
new_on = 'song_id'
# new_on = 'source_type_guess'

print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('<'*20)

print('inspecting:', on)
print('>'*20)
print('any null:', df[on].isnull().values.any())
print('null number:', df[on].isnull().values.sum())
print(on, 'dtype:', df[on].dtypes)
print('describing', on, ':')
print(df[on].describe())
print('<'*20)
l = df[on]
s = set(l)
print('list len:', len(l))
print('set len:', len(s))
# print(s)
# dff = df[df['source_system_tab' == 'unknown']]
# print('---===', len(dff))
# print('<'*20)
# ddd = df.dtypes.to_dict()
# for i in ddd:
#     on = i
#     print('inspecting:', on)
#     print('>' * 20)
#     print('any null:', df[on].isnull().values.any())
#     print('null number:', df[on].isnull().values.sum())
#     print('<'*20)
#     print()

# plot = True
plot = False
if plot:
    plt.figure(figsize=(15, 12))
    # dff = pd.DataFrame()
    # dff[on] = df[on].dropna()
    # sns.distplot(df[on])
    sns.countplot(df[on])
    # plt.xlim((0, 0.3))
    plt.show()


isit = True
# isit = False
if isit:

    print('/' * 30)
    print('/' * 30)
    print('/' * 30)
    print('inspecting:', new_on)
    print('>' * 20)
    print('any null:', df[new_on].isnull().values.any())
    print('null number:', df[new_on].isnull().values.sum())
    print(new_on, 'dtype:', df[new_on].dtypes)
    print('describing', new_on, ':')
    print(df[new_on].describe())
    print('<' * 20)
    li = df[new_on]
    s = set(li)
    print('after, df len:', len(df))
    print('list len:', len(li))
    print('set len:', len(s))
    # print(s)
    # plot = True
    plot = False
    if plot:
        plt.figure(figsize=(15, 12))
        # dff = pd.DataFrame()
        # dff[new_on] = df[new_on].dropna()
        # del df
        # sns.distplot(df[new_on])
        sns.countplot(df[new_on])

        plt.show()
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


