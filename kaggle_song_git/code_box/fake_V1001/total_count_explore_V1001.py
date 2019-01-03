import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
# save_dir = '../fake/'
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

load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))
print('<'*20)
# save_dir = '../fake/'
# # df = df[df['target'] == 1]
# # df = df[df['target'] == 0]
#
# count = {k: v for k, v in df['song_id'].value_counts().iteritems()}
# pickle.dump(count, open(save_dir+'total_song_count_dict.save', "wb"))
# # pickle.dump(count, open(save_dir+'liked_song_count_dict.save', "wb"))
# # pickle.dump(count, open(save_dir+'disliked_song_count_dict.save', "wb"))
# del count


kinds = {}


def genre_id_count(x):
    global kinds
    if x == 'no_genre_id':
        return 0
    else:
        a = x.count('|') + 1
        if a == 1:
            if x in kinds:
                pass
            else:
                kinds[x] = 0
            kinds[x] += 1
        else:
            for i in x.split('|'):
                if i in kinds:
                    pass
                else:
                    kinds[i] = 0
                kinds[i] += 1
        return a


df['fake_genre_type_count'] = df['genre_ids'].apply(genre_id_count).astype(np.int64)

print('kinds len:', len(kinds))
# lists = sorted(kinds.items()) # sorted by key, return a list of tuples
# x, y = zip(*lists)
# for i in kinds:
#     print(i, ':', kinds[i])
import operator
# x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
# sorted_x = sorted(kinds.items(), key=operator.itemgetter(1))
# sorted_x = reversed(sorted_x)
# x, y = zip(*sorted_x)
# print(type(sorted_x))
# print(sorted_x)

storage = storage = '../fake/'
# kinds = pickle.load(open(storage + 'total_song_count_dict.save', "rb"))
# df['fake_song_count'] = df['song_id'].apply(get_count1).astype(np.int64)
# kinds = pickle.load(open(storage + 'total_artist_count_dict.save', "rb"))
kinds = pickle.load(open(storage + 'total_member_count_dict.save', "rb"))
sorted_x = sorted(kinds.items(), key=operator.itemgetter(1))
sorted_x = reversed(sorted_x)
x, y = zip(*sorted_x)
plot = True
# plot = False
if plot:
    plt.figure(figsize=(15, 12))
    # dff = pd.DataFrame()
    # dff[on] = df[on].dropna()
    # sns.distplot(df[on])
    # sns.countplot(kinds)
    # a = range(50)
    b = len(y)
    # b = 200
    # print('a len:', len(a))
    # print('a type:', type(a))
    # print('b len:', len(b))
    # print('b type:', type(b))
    plt.plot([i for i in range(b)], y[:b])
    # plt.xlabel(x[:50])
    # sns.countplot(df[df['fake_genre_type_count'] > 3]['fake_genre_type_count'])
    # plt.xlim((0, 0.3))
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

