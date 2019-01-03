import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'


df = pd.read_csv(save_dir+"song.csv",
                 dtype={'song_id': 'category',
                        'song_length': np.float64,
                        'genre_ids': 'category',
                        'artist_name': 'category',
                        'composer': 'category',
                        'lyricist': 'category',
                        'language': np.float32,
                        'name': 'category',
                        'isrc': 'category'
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

def isrc_to_year_fre(x):
    if x == 'MISSING':
        return 2016
    else:
        if int(x[5:7]) > 17:
            return 1900 + int(x[5:7])
        else:
            return 2000 + int(x[5:7])


def isrc_to_year(x):
    if x == 'MISSING':
        return 0
    else:
        if int(x[5:7]) > 17:
            return 1900 + int(x[5:7])
        else:
            return 2000 + int(x[5:7])


def isrc_to_c_fre(x):
    if x == 'MISSING':
        return 'US'
    else:
        return x[0:2]


def isrc_to_c(x):
    if x == 'MISSING':
        return 'MISSING'
    else:
        return x[0:2]


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


def isrc_to_rc(x):
    if x == 'MISSING':
        return 'MISSING'
    else:
        return x[2:5]


def rest_rc(x):
    if x == 'MISSING':
        return 'MISSING'
    else:
        return x[-5:]

# df['sn'] = df.index


def genre_ids_fre(x):
    if x == 'unknown':
        return '465'
    else:
        return x


df['name'] = df['name'].astype(object)
df['name'].fillna('MISSING', inplace=True)

df['lyricist'] = df['lyricist'].astype(object)
df['lyricist'].fillna('MISSING', inplace=True)

df['composer'] = df['composer'].astype(object)
df['composer'].fillna('MISSING', inplace=True)

df['genre_ids'] = df['genre_ids'].astype(object)
df['genre_ids'].fillna('MISSING', inplace=True)
# df['genre_ids_fre_song'] = df['genre_ids'].apply(genre_ids_fre).astype('category')

df['artist_name'] = df['artist_name'].astype(object)
df['artist_name'].fillna('MISSING', inplace=True)

# df['fake_genre_type_count'] = df['genre_ids'].apply(genre_id_count).astype(np.int64)

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
df['isrc'] = df['isrc'].astype(object)
df['isrc'].fillna('MISSING', inplace=True)

# df['song_year_fre_song'] = df['isrc'].apply(isrc_to_year_fre).astype(np.int64)
# df['song_year_fre_song'] = df['song_year_fre_song'].astype('category')

df['song_year'] = df['isrc'].apply(isrc_to_year).astype(np.int64)
# df['song_year'] = df['song_year'].astype('category')

# df['song_year_bin_range'] = df['song_year'].apply(song_year_bin_range).astype(np.int64)
# df['song_year_chunk_range'] = df['song_year'].apply(song_year_chunk_range).astype(np.int64)
# df['song_country_fre_song'] = df['isrc'].apply(isrc_to_c_fre).astype('category')
df['song_country'] = df['isrc'].apply(isrc_to_c).astype('category')

df['rc'] = df['isrc'].apply(isrc_to_rc).astype('category')
df['isrc_rest'] = df['isrc'].apply(rest_rc).astype('category')


# df['artist_composer'] = (df['artist_name'] == df['composer']).astype(np.int8)
# df['artist_composer_lyricist'] = ((df['artist_name'] == df['composer']) &
#                                   (df['artist_name'] == df['lyricist']) &
#                                   (df['composer'] == df['lyricist'])).astype(np.int8)

kinds = {'0': 0}


def genre_id_count(x):
    global kinds
    if x == 'MISSING':
        kinds['0'] += 1
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
# for k in kinds:
#     print(k, ':', kinds[k])


def top1(x):
    if x == 'MISSING':
        return '0'
    else:
        a = x.count('|') + 1
        if a == 1:
            return x
        else:
            top_1 = '0'
            best_1 = 0
            top_2 = '0'
            best_2 = 0
            top_3 = '0'
            best_3 = 0
            for g in x.split('|'):
                if g in kinds:
                    if kinds[g] > best_1:
                        top_3 = top_2
                        best_3 = best_2
                        top_2 = top_1
                        best_2 = best_1

                        top_1 = g
                        best_1 = kinds[g]

                    elif kinds[g] > best_2:
                        top_3 = top_2
                        best_3 = best_2
                        top_2 = g
                        best_2 = kinds[g]
                    elif kinds[g] > best_3:
                        top_3 = g
                        best_3 = kinds[g]
                    else:
                        pass
            return top_1


def top2(x):
    if x == 'MISSING':
        return '0'
    else:
        a = x.count('|') + 1
        if a == 1:
            return x
        else:
            top_1 = '0'
            best_1 = 0
            top_2 = '0'
            best_2 = 0
            top_3 = '0'
            best_3 = 0
            for g in x.split('|'):
                if g in kinds:
                    if kinds[g] > best_1:
                        top_3 = top_2
                        best_3 = best_2
                        top_2 = top_1
                        best_2 = best_1

                        top_1 = g
                        best_1 = kinds[g]

                    elif kinds[g] > best_2:
                        top_3 = top_2
                        best_3 = best_2
                        top_2 = g
                        best_2 = kinds[g]
                    elif kinds[g] > best_3:
                        top_3 = g
                        best_3 = kinds[g]
                    else:
                        pass
            return top_2


def top3(x):
    if x == 'MISSING':
        return '0'
    else:
        a = x.count('|') + 1
        if a == 1:
            return x
        else:
            top_1 = '0'
            best_1 = 0
            top_2 = '0'
            best_2 = 0
            top_3 = '0'
            best_3 = 0
            for g in x.split('|'):
                if g in kinds:
                    if kinds[g] > best_1:
                        top_3 = top_2
                        best_3 = best_2
                        top_2 = top_1
                        best_2 = best_1

                        top_1 = g
                        best_1 = kinds[g]

                    elif kinds[g] > best_2:
                        top_3 = top_2
                        best_3 = best_2
                        top_2 = g
                        best_2 = kinds[g]
                    elif kinds[g] > best_3:
                        top_3 = g
                        best_3 = kinds[g]
                    else:
                        pass
            return top_3


df['top1_in_song'] = df['genre_ids'].apply(top1).astype('category')
df['top2_in_song'] = df['genre_ids'].apply(top2).astype('category')
df['top3_in_song'] = df['genre_ids'].apply(top3).astype('category')


counter = {}

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


def get_count(x):
    try:
        return counter[x]
    except KeyError:
        return 1


def count_up(on):
    global counter, df
    print()
    print('>'*30)
    print()
    print('working on:', on)
    counter = {k: v for k, v in df[on].value_counts().iteritems()}
    df['ISC_'+on] = df[on].apply(get_count).astype(np.int64)
    if 'MISSING' in counter:
        print('MISSING:', counter['MISSING'])
        counter['MISSING'] = 0
        df['ISCZ_'+on] = df[on].apply(get_count).astype(np.int64)
    print('counter len:', len(counter))
    # for i in counter:
    #     print(i, ':', counter[i])
    del counter
    print()
    print('<' * 30)


def count_up_year(on):
    global counter, df
    print()
    print('>'*30)
    print()
    print('working on:', on)
    counter = {k: v for k, v in df[on].value_counts().iteritems()}
    df['ISC_'+on] = df[on].apply(get_count).astype(np.int64)
    counter[0] = 0
    df['ISCZ_'+on] = df[on].apply(get_count).astype(np.int64)
    print('counter len:', len(counter))
    # for i in counter:
    #     print(i, ':', counter[i])
    del counter
    print()
    print('<' * 30)


# count_up('')
count_up('genre_ids')
count_up('top1_in_song')
count_up('top2_in_song')
count_up('top3_in_song')

count_up('artist_name')
count_up('composer')
count_up('lyricist')
count_up('name')
count_up('language')

count_up('isrc')
count_up('song_country')
count_up('rc')
count_up('isrc_rest')
count_up_year('song_year')
# count_up('')
# count_up('')

# df.drop('genre_ids', axis=1, inplace=True)
# df.drop(['isrc', ], axis=1, inplace=True)
df.drop(
    [
        'genre_ids',
        'artist_name',
        'lyricist',
        'composer',
        'language',
        'name',
        'isrc',
        'song_country',
        'rc',
        'isrc_rest',
        'top1_in_song',
        'top2_in_song',
        'top3_in_song',
        'fake_genre_type_count'

    ],
    axis=1, inplace=True
)


def log10me(x):
    return np.log10(x)


def log2me(x):
    return np.log2(x)


def lnme(x):
    return np.log(x)


ccc = [i for i in df.columns]
print(type(ccc))
ccc.remove('song_id')
for col in df.columns:
    if not col == 'song_id':
        df[col+'_log10'] = df[col].apply(log10me).astype(np.float64)
        # df[col + '_log2'] = df[col].apply(log2me).astype(np.float64)
        # df[col + '_ln'] = df[col].apply(lnme).astype(np.float64)
df.drop(ccc, axis=1, inplace=True)


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
    print()
    print(on, 'dtype:', df[on].dtypes)
    print('describing', on, ':')
    print(df[on].describe())
    print('<'*20)
    l = df[on]
    s = set(l)
    print('list len:', len(l))
    print('set len:', len(s))
    print()
print('<'*20)
print('<'*20)
print('<'*20)



print('creating custom member.')
save_name = 'custom_song_'
vers = 'fixed'
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

print('p2 fixup custom song done.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

