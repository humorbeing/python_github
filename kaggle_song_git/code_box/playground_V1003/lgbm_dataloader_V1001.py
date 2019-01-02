import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time


since = time.time()

print('Loading data...')
data_path = '../data/'
train = pd.read_csv(data_path + 'train.csv',
                    dtype={'msno': 'category',
                           'source_system_tab': 'category',
                           'source_screen_name': 'category',
                           'source_type': 'category',
                           'target': np.uint8,
                           'song_id': 'category'
                           }
                    )
test = pd.read_csv(data_path + 'test.csv',
                   dtype={'msno': 'category',
                          'source_system_tab': 'category',
                          'source_screen_name': 'category',
                          'source_type': 'category',
                          'song_id': 'category'
                          }
                   )
songs = pd.read_csv(data_path + 'songs.csv',
                    dtype={'genre_ids': 'category',
                           'language': 'category',
                           'artist_name': 'category',
                           'composer': 'category',
                           'lyricist': 'category',
                           'song_id': 'category',
                           'song_length': np.uint64
                           }
                    )
members = pd.read_csv(data_path + 'members.csv',
                      dtype={'city': np.uint8,
                             'bd': np.uint8,
                             'gender': 'category',
                             'registered_via': np.uint8,
                             'msno': 'category'
                             },
                      parse_dates=['registration_init_time',
                                   'expiration_date']
                      )
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv',
                          dtype={'song_id': 'category',
                                 'name': 'category',
                                 'isrc': 'category'
                                 }
                          )
print('Done loading...')

print('Data merging...')

train = train.merge(songs, on='song_id', how='left')
test = test.merge(songs, on='song_id', how='left')


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan


songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on='song_id', how='left')
train.song_length.fillna(200000, inplace=True)
train.song_length = train.song_length.astype(np.uint32)
train.song_id = train.song_id.astype('category')

test = test.merge(songs_extra, on='song_id', how='left')
test.song_length.fillna(200000, inplace=True)
test.song_length = test.song_length.astype(np.uint32)
test.song_id = test.song_id.astype('category')

# import gc
del members, songs, songs_extra

print('Done merging...')

print("Adding new features")


members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)

members['registration_year'] = members['registration_init_time'].dt.year
members['registration_month'] = members['registration_init_time'].dt.month
members['registration_date'] = members['registration_init_time'].dt.day

members['expiration_year'] = members['expiration_date'].dt.year
members['expiration_month'] = members['expiration_date'].dt.month
members['expiration_date'] = members['expiration_date'].dt.day
members = members.drop(['registration_init_time'], axis=1)


def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1


train['genre_ids'].cat.add_categories(['no_genre_id']).fillna('no_genre_id', inplace=True)
test['genre_ids'].cat.add_categories(['no_genre_id']).fillna('no_genre_id', inplace=True)
train['genre_ids_count'] = train['genre_ids'].apply(genre_id_count).astype(np.int8)
test['genre_ids_count'] = test['genre_ids'].apply(genre_id_count).astype(np.int8)


def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    # return sum(map(x.count, ['|', '/', '\\', ';']))


train['lyricist'].cat.add_categories(['no_lyricist']).fillna('no_lyricist', inplace=True)
test['lyricist'].cat.add_categories(['no_lyricist']).fillna('no_lyricist', inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)


def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1


train['composer'].cat.add_categories(['no_composer']).fillna('no_composer', inplace=True)
test['composer'].cat.add_categories(['no_composer']).fillna('no_composer', inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)


def is_featured(x):
    if 'feat' in str(x):
        return 1
    return 0


# train['artist_name'].cat.add_categories(['no_artist']).fillna('no_artist', inplace=True)
# test['artist_name'].cat.add_categories(['no_artist']).fillna('no_artist', inplace=True)
train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int8)


def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')


train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int8)
test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)

# if artist is same as composer
# train['artist_composer'] = (train['artist_name'] == train['composer']).astype(np.int8)
# test['artist_composer'] = (test['artist_name'] == test['composer']).astype(np.int8)

# if artist, lyricist and composer are all three same
# train['artist_composer_lyricist'] = (
# (train['artist_name'] == train['composer']) & (train['artist_name'] == train['lyricist']) & (
# train['composer'] == train['lyricist'])).astype(np.int8)
# test['artist_composer_lyricist'] = (
# (test['artist_name'] == test['composer']) & (test['artist_name'] == test['lyricist']) & (
# test['composer'] == test['lyricist'])).astype(np.int8)


# is song language 17 or 45.
def song_lang_boolean(x):
    if '17.0' in str(x) or '45.0' in str(x):
        return 1
    return 0


train['language'].fillna('31.0', inplace=True)
test['language'].fillna('31.0', inplace=True)
train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)

_mean_song_length = np.mean(train['song_length'])


def smaller_song(x):
    if x < _mean_song_length:
        return 1
    return 0


train['smaller_song'] = train['song_length'].apply(smaller_song).astype(np.int8)
test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int8)

# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
_dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}


def count_song_played(x):
    try:
        return _dict_count_song_played_train[x]
    except KeyError:
        try:
            return _dict_count_song_played_test[x]
        except KeyError:
            return 0


train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)

# number of times the artist has been played
_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
_dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}


def count_artist_played(x):
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
        try:
            return _dict_count_artist_played_test[x]
        except KeyError:
            return 0


train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64)

print(train.head())
print(test.head())


train.to_csv('train_merge_feature.csv', index=False)
test.to_csv('test_merge_feature.csv', index=False)

print("Done adding features")

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 "/media/ray/PNU@myPC@DDDDD/workspace/python/projects/big data kaggle/playground_V1002/fkk_feature_engi_Lgbm.py"
Loading data...
Done loading...
Data merging...
Done merging...
Adding new features
                                           msno  \
0  FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=   
1  Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=   
2  Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=   
3  Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=   
4  FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=   

                                        song_id source_system_tab  \
0  BBzumQNXUHKdEBOB7mAJuzok+IJA1c2Ryg/yzTF6tik=           explore   
1  bhp/MpSNoqoxOIB+/l8WPqu6jldth4DIpCm3ayXnJqM=        my library   
2  JNWfrrC7zNN7BdMpsISKa4Mw+xVJYNnxXh3/Epw7QgY=        my library   
3  2A87tzfnJTSWqD7gIZHisolhe4DMdzkbd6LzO1KHjNs=        my library   
4  3qm6XTZ6MOCU11x8FIVbAGH5l5uMkT3/ZalWG1oo2Gc=           explore   

    source_screen_name      source_type  target  song_length genre_ids  \
0              Explore  online-playlist       1       206471       359   
1  Local playlist more   local-playlist       1       284584      1259   
2  Local playlist more   local-playlist       1       225396      1259   
3  Local playlist more   local-playlist       1       255512      1019   
4              Explore  online-playlist       1       187802      1011   

       artist_name                                 composer  \
0         Bastille                     Dan Smith| Mark Crew   
1  Various Artists                                      NaN   
2              Nas     N. Jones、W. Adams、J. Lordan、D. Ingle   
3         Soundway                            Kwadwo Donkoh   
4      Brett Young  Brett Young| Kelly Archer| Justin Ebach   

          ...          song_year genre_ids_count lyricists_count  \
0         ...             2016.0               1               1   
1         ...             1999.0               1               1   
2         ...             2006.0               1               1   
3         ...             2010.0               1               1   
4         ...             2016.0               1               1   

   composer_count is_featured artist_count  song_lang_boolean  smaller_song  \
0               2           0            0                  0             1   
1               1           0            0                  0             0   
2               1           0            0                  0             1   
3               1           0            0                  0             0   
4               3           0            0                  0             1   

   count_song_played  count_artist_played  
0                215                 1140  
1                  1               303616  
2                  4                  289  
3                  1                    1  
4                412                  427  

[5 rows x 33 columns]
   id                                          msno  \
0   0  V8ruy7SGk7tDm3zA51DPpn6qutt+vmKMBKa21dp54uM=   
1   1  V8ruy7SGk7tDm3zA51DPpn6qutt+vmKMBKa21dp54uM=   
2   2  /uQAlrAkaczV+nWCd2sPF2ekvXPRipV7q0l+gbLuxjw=   
3   3  1a6oo/iXKatxQx4eS9zTVD+KlSVaAFbTIqVvwLC1Y0k=   
4   4  1a6oo/iXKatxQx4eS9zTVD+KlSVaAFbTIqVvwLC1Y0k=   

                                        song_id source_system_tab  \
0  WmHKgKMlp1lQMecNdNvDMkvIycZYHnFwDT72I5sIssc=        my library   
1  y/rsZ9DC7FwK5F2PK2D5mj+aOBUJAjuu3dZ14NgE0vM=        my library   
2  8eZLFOdGVdXBSqoAv5nsLigeH2BvKXzTQYtUM53I0k4=          discover   
3  ztCf8thYsS4YN3GcIL/bvoxLm/T5mYBVKOO4C9NiVfQ=             radio   
4  MKVMpslKcQhMaFEgcEQhEfi5+RZhMYlU3eRDpySrH8Y=             radio   

    source_screen_name          source_type  song_length genre_ids  \
0  Local playlist more        local-library       224130       458   
1  Local playlist more        local-library       320470       465   
2                  NaN  song-based-playlist       315899      2022   
3                Radio                radio       285210       465   
4                Radio                radio       197590       873   

          artist_name                                   composer  \
0  梁文音 (Rachel Liang)                             Qi Zheng Zhang   
1        林俊傑 (JJ Lin)                                        林俊傑   
2  Yu Takahashi (高橋優)                               Yu Takahashi   
3                  U2  The Edge| Adam Clayton| Larry Mullen| Jr.   
4       Yoga Mr Sound                                Neuromancer   

          ...          song_year genre_ids_count lyricists_count  \
0         ...             2014.0               1               1   
1         ...             2010.0               1               2   
2         ...             2010.0               1               1   
3         ...             2002.0               1               1   
4         ...             2011.0               1               1   

   composer_count is_featured artist_count  song_lang_boolean  smaller_song  \
0               1           0            0                  0             1   
1               1           0            0                  0             0   
2               1           0            0                  1             0   
3               4           0            0                  0             0   
4               1           0            0                  0             1   

   count_song_played  count_artist_played  
0                694                13654  
1               6090               115325  
2                  5                  989  
3                 31                  698  
4                  5                  180  

[5 rows x 33 columns]
Done adding features

[timer]: complete in 15m 50s

Process finished with exit code 0
'''