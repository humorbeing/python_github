import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_fillna3'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
# df['source_system_tab'] = df['source_system_tab'].astype(object)
# df['source_system_tab'].fillna('my library', inplace=True)
# df['source_screen_name'] = df['source_screen_name'].astype(object)
# df['source_screen_name'].fillna('Local playlist more', inplace=True)
# df['source_type'] = df['source_type'].astype(object)
# df['source_type'].fillna('local-library', inplace=True)

# df = df[df['target'] == 1]
df = df[df['target'] == 0]

song_count = {k: v for k, v in df['song_id'].value_counts().iteritems()}
# pickle.dump(song_count, open(save_dir+'song_count_dict.save', "wb"))
# pickle.dump(song_count, open(save_dir+'liked_song_count_dict.save', "wb"))
pickle.dump(song_count, open(save_dir+'disliked_song_count_dict.save', "wb"))
del song_count
# def count_song_played(x):
#     try:
#         return _dict_count_song_played_train[x]
#     except KeyError:
#         try:
#             return _dict_count_song_played_test[x]
#         except KeyError:
#             return 0


# def count_song_played(x):
#     try:
#         return song_count[x]
#     except KeyError:
#         return 0
#
#
# # def count_song_played(x):
# #     return song_count[x]
#
#
# load_name = 'train_fillna3'
# dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
# df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
# del dt
# df['unliked_count_song_played'] = df['song_id'].apply(count_song_played).astype(np.int64)

artist_count = {k: v for k, v in df['artist_name'].value_counts().iteritems()}
# pickle.dump(artist_count, open(save_dir+'artist_count_dict.save', "wb"))
# pickle.dump(artist_count, open(save_dir+'liked_artist_count_dict.save', "wb"))
pickle.dump(artist_count, open(save_dir+'disliked_artist_count_dict.save', "wb"))
del artist_count
# def count_artist_played(x):
#     return count_artist[x]
#
#
# df['count_song_played'] = df['song_id'].apply(count_song_played).astype(np.int64)
# df['count_artist_played'] = df['artist_name'].apply(count_artist_played).astype(np.int64)


on = False
# on = 'target'
# on = 'source_system_tab'
# on = 'source_screen_name'
# on = 'source_type'
# on = 'count_song_played'
# on = 'count_artist_played'
# on = 'unliked_count_song_played'


print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('<'*20)
if on:
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
    # # print(s)
    print('<'*20)
# check_all = True
check_all = False
if check_all:
    ddd = df.dtypes.to_dict()
    for i in ddd:
        on = i
        print('inspecting:', on)
        print('>' * 20)
        print('any null:', df[on].isnull().values.any())
        print('null number:', df[on].isnull().values.sum())
        print('<'*20)
        print()

# plot = True
plot = False
if plot:
    plt.figure(figsize=(15, 12))
    # dff = pd.DataFrame()
    # dff[on] = df[on].dropna()
    # del df
    sns.distplot(df[on])
    # sns.countplot(df[on])
    plt.show()
# _dict_count_song_played_train = {k: v for k, v in df['song_id'].value_counts().iteritems()}
# for i in _dict_count_song_played_train:
#     print(i, ':', _dict_count_song_played_train[i])

plt.show()
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

