import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'


load_name = 'custom_song_fixed'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)


# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
count = {}
# count1 = {}
# count2 = {}


def get_count1(x):
    try:
        return count[x]
    except KeyError:
        return 1


def get_count(x):
    try:
        return count[x]
    except KeyError:
        return 0

# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


storage = '../fake/'
count = pickle.load(open(storage + 'total_song_count_dict.save', "rb"))
df['fake_song_count'] = df['song_id'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(storage + 'liked_song_count_dict.save', "rb"))
# df['fake_liked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(storage + 'disliked_song_count_dict.save', "rb"))
# df['fake_disliked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
count = pickle.load(open(storage + 'total_artist_count_dict.save', "rb"))
df['fake_artist_count'] = df['artist_name'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(storage + 'liked_artist_count_dict.save', "rb"))
# df['fake_liked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
# count = pickle.load(open(storage + 'disliked_artist_count_dict.save', "rb"))
# df['fake_disliked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)


count = pickle.load(open(storage + 'total_language_count_dict.save', "rb"))
# df['fake_language_count'] = df['language'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(storage + 'liked_member_count_dict.save', "rb"))
# df['fake_liked_member_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(storage + 'disliked_member_count_dict.save', "rb"))
# df['fake_disliked_member_count'] = df['song_id'].apply(get_count).astype(np.int64)
del count


count = pickle.load(open(storage + 'total_genre_ids_count_dict.save', "rb"))
# df['fake_genre_ids_count'] = df['genre_ids'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(storage + 'liked_member_count_dict.save', "rb"))
# df['fake_liked_member_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(storage + 'disliked_member_count_dict.save', "rb"))
# df['fake_disliked_member_count'] = df['song_id'].apply(get_count).astype(np.int64)
del count
count = pickle.load(open(storage + 'total_song_year_count_dict.save', "rb"))
df['fake_song_year_count'] = df['song_year'].apply(get_count1).astype(np.int64)
count = pickle.load(open(storage + 'total_song_country_count_dict.save', "rb"))
df['fake_song_country_count'] = df['song_country'].apply(get_count1).astype(np.int64)
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

count1 = {}
count2 = {}


def c1_c2(x):
    try:
        c1 = count1[x]
    except KeyError:
        c1 = 0
    try:
        c2 = count2[x]
    except KeyError:
        c2 = 1

    return c1 / c2


def c1_c2_song(x):
    try:
        c1 = count1[x]
    except KeyError:
        c1 = 0
    try:
        c2 = count2[x]
    except KeyError:
        c2 = 0

    if c1 == 0 and c2 == 0:
        return 0.61
    else:
        if c2 == 0:
            return 100 * c1
        else:
            return c1 / c2


# def c1_c2_member(x):
#     try:
#         c1 = count1[x]
#     except KeyError:
#         c1 = 0
#     try:
#         c2 = count2[x]
#     except KeyError:
#         c2 = 0
#
#     if c1 == 0 and c2 == 0:
#         return 0
#     else:
#         if c2 == 0:
#             return 100 * c1
#         else:
#             return c1 / c2
#
#
# count1 = pickle.load(open(save_dir + storage + 'liked_song_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'song_count_dict.save', "rb"))
# df['like_song_chance'] = df['song_id'].apply(c1_c2).astype(np.float16)
#
# count1 = pickle.load(open(save_dir + storage + 'disliked_song_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'song_count_dict.save', "rb"))
# df['dislike_song_chance'] = df['song_id'].apply(c1_c2).astype(np.float16)
#
# count1 = pickle.load(open(save_dir + storage + 'liked_song_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'disliked_song_count_dict.save', "rb"))
# df['song_like_dislike'] = df['song_id'].apply(c1_c2_song).astype(np.float16)
#
# ###
#
# count1 = pickle.load(open(save_dir + storage + 'liked_artist_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'artist_count_dict.save', "rb"))
# df['like_artist_chance'] = df['artist_name'].apply(c1_c2).astype(np.float16)
#
# count1 = pickle.load(open(save_dir + storage + 'disliked_artist_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'artist_count_dict.save', "rb"))
# df['dislike_artist_chance'] = df['artist_name'].apply(c1_c2).astype(np.float16)
#
# count1 = pickle.load(open(save_dir + storage + 'liked_artist_count_dict.save', "rb"))
# count2 = pickle.load(open(save_dir + storage + 'disliked_artist_count_dict.save', "rb"))
# df['artist_like_dislike'] = df['artist_name'].apply(c1_c2).astype(np.float16)


kinds = pickle.load(open(storage + 'total_single_genre_dict.save', "rb"))
print('kinds len:', len(kinds))


def top1(x):
    if x == 'unknown':
        return '0'
    else:
        a = x.count('|') + 1
        if a == 1:
            return x
        else:
            top_1 = '0'
            best_1 = 0
            for g in x.split('|'):
                if g in kinds:
                    if kinds[g] > best_1:
                        top_1 = g
                        best_1 = kinds[g]
            return top_1


df['fake_top1'] = df['genre_ids'].apply(top1).astype('category')


def top1_count(x):
    if x in kinds:
        return kinds[x]
    else:
        return 0


df['fake_top1_count'] = df['fake_top1'].apply(top1_count).astype(np.int64)


# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


print('creating custom member.')
save_name = 'custom_song_'
vers = 'fixed'
d = df.dtypes.to_dict()

print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)
pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

print('p2 fake attach done.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

