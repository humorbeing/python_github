import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'


df = pd.read_csv(save_dir+"custom_song_data.csv",
                 dtype={'song_id': 'category',
                        'song_length': np.float64,
                        'genre_ids': 'category',
                        'artist_name': 'category',
                        'composer': 'category',
                        'lyricist': 'category',
                        'language': np.float32,
                        'name': 'category',
                        # 'isrc': 'category'
                        }
                 )


def fix_language(x):
    if x == -1.0:
        return 1
    elif x == 3.0:
        return 2
    elif x == 10.0:
        return 3
    elif x == 17.0:
        return 4
    elif x == 24.0:
        return 5
    elif x == 31.0:
        return 6
    elif x == 38.0:
        return 7
    elif x == 45.0:
        return 8
    elif x == 52.0:
        return 9
    elif x == 59.0:
        return 10
    else:
        return 6


def length_range(x):
    n = 5000
    split = np.linspace(185, 12174000, n)
    its = 1
    while True:
        if its == n:
            return 0
        else:
            if x < int(split[its]):
                return its
            else:
                its += 1


def length_bin_range(x):
    if x < 500000:
        return 1
    else:
        return 0


def length_chunk_range(x):

    n = 10
    split = np.linspace(185, 500000, n)
    its = 1
    while True:
        if its == n:
            return 0
        else:
            if x < int(split[its]):
                return its
            else:
                its += 1


# def isrc_to_year(isrc):
#     if type(isrc) == str:
#         if int(isrc[5:7]) > 17:
#             return 1900 + int(isrc[5:7])
#         else:
#             return 2000 + int(isrc[5:7])
#     else:
#         # print('here')
#         a = np.random.poisson(2016, 500)
#         a = int(np.mean(a))
#         if a > 2016:
#             a = int(np.random.uniform(1918, 2017))
#         return a


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return 0


def song_year_bin_range(x):
    if x < 1980:
        return 0
    else:
        return 1


def song_year_chunk_range(x):
    if x < 1999:
        return 0
    else:
        return x


def isrc_to_c(isrc):
    if type(isrc) == str:
        return isrc[0:2]
    else:
        return '0'


def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        a = sum(map(x.count, ['|', '/', '\\', ';'])) + 1
        # if a > 1:
        #     print(x)
        return a


def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1


def genre_id_count(x):
    # global kinds
    if x == 'unknown':
        return 0
    else:
        a = x.count('|') + 1
        # if a == 1:
        #     kinds.add(x)
        # else:
        #     for i in x.split('|'):
        #         kinds.add(i)
        return a


def isrc_to_rc(isrc):
    if type(isrc) == str:
        return isrc[2:5]
    else:
        return 'no_rc'


# df['sn'] = df.index

# df['lyricist'] = df['lyricist'].astype(object)
# df['lyricist'].fillna('no_lyricist', inplace=True)
# df['composer'] = df['composer'].astype(object)
# df['composer'].fillna('no_composer', inplace=True)
df['genre_ids'] = df['genre_ids'].astype(object)
df['genre_ids'].fillna('unknown', inplace=True)
df['artist_name'] = df['artist_name'].astype(object)
df['artist_name'].fillna('no_artist_name', inplace=True)

df['fake_genre_type_count'] = df['genre_ids'].apply(genre_id_count).astype(np.int64)

# df['lyricists_count'] = df['lyricist'].apply(lyricist_count).astype(np.int8)
# df['composer_count'] = df['composer'].apply(composer_count).astype(np.int8)
# df['genre_ids_count'] = df['genre_ids'].apply(genre_id_count).astype(np.int8)


df['language'] = df['language'].apply(fix_language).astype(np.int8)
df['language'] = df['language'].astype('category')

# df.song_length.fillna(248770, inplace=True)
# df['song_length'] = df['song_length'].astype(np.int64)
# df['length_range'] = df['song_length'].apply(length_range).astype(np.int64)
# df['length_bin_range'] = df['song_length'].apply(length_bin_range).astype(np.int64)
# df['length_bin_range'] = df['length_bin_range'].astype('category')
# df['length_chunk_range'] = df['song_length'].apply(length_chunk_range).astype(np.int64)
# df['length_chunk_range'] = df['length_chunk_range'].astype('category')
df['song_year'] = df['isrc'].apply(isrc_to_year).astype(np.int64)
# df['song_year_bin_range'] = df['song_year'].apply(song_year_bin_range).astype(np.int64)
# df['song_year_chunk_range'] = df['song_year'].apply(song_year_chunk_range).astype(np.int64)
df['song_country'] = df['isrc'].apply(isrc_to_c).astype(object)
# df['rc'] = df['isrc'].apply(isrc_to_rc).astype(object)



# df['artist_composer'] = (df['artist_name'] == df['composer']).astype(np.int8)
# df['artist_composer_lyricist'] = ((df['artist_name'] == df['composer']) &
#                                   (df['artist_name'] == df['lyricist']) &
#                                   (df['composer'] == df['lyricist'])).astype(np.int8)

count = {}


def get_count(x):
    try:
        return count[x]
    except KeyError:
        return 0


# storage = 'storage/'
# count = pickle.load(open(save_dir + storage + 'song_count_dict.save', "rb"))
# df['song_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + storage + 'liked_song_count_dict.save', "rb"))
# df['liked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + storage + 'disliked_song_count_dict.save', "rb"))
# df['disliked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
#
# count = pickle.load(open(save_dir + storage + 'artist_count_dict.save', "rb"))
# df['artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + storage + 'liked_artist_count_dict.save', "rb"))
# df['liked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + storage + 'disliked_artist_countt_dict.save', "rb"))
# df['disliked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
# del count
df.drop(['isrc', 'name'], axis=1, inplace=True)
df.drop(['lyricist',
         'composer',
         # 'genre_ids',
         'song_length',
         ],
        axis=1, inplace=True)


print('creating custom member.')
save_name = 'custom_song_'
vers = 'fixed'
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

print('p2 fixup custom song done.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

