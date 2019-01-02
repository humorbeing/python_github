import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
# df_train = pd.read_csv(data_dir+"train.csv")
#
# df_songs = pd.read_csv(data_dir+"songs.csv")
# # #
# df_songs_extra = pd.read_csv(data_dir+"song_extra_info.csv")

# df = pd.read_csv(save_dir+"song_merge.csv",
# df = pd.read_csv(save_dir+"custom_song_data.csv",
#                  dtype={'song_id': 'category',
#                         'song_length': np.float64,
#                         'genre_ids': 'category',
#                         'artist_name': 'category',
#                         'composer': 'category',
#                         'lyricist': 'category',
#                         'language': np.float32,
#                         'name': 'category',
#                         'isrc': 'category'
#                         }
#                  )
# save_dir = '../saves/'
load_name = 'CSF_fixed'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv",
                 dtype=dt)
del dt
# df_test = pd.read_csv(data_dir+"test.csv")

# df_song_data = pd.read_csv(save_dir+'song_data.csv')

# df_custom_song_data = pd.read_csv(save_dir+'custom_song_data.csv')
# df_custom_members = pd.read_csv(save_dir+'custom_members.csv')

# print(len(df_songs))
# print(len(df_songs_extra))
# df = df_songs.merge(df_songs_extra, on='song_id', how='left')
# del df_songs_extra, df_songs
# print(len(df))
# print(df.head())
# # print(df.describe())
# print(df.dtypes)
# df.to_csv(save_dir+'song_merge.csv', index=False)
# df = df_train
# df = df_test

# on = 'target'
# on = 'source_system_tab'
# on = 'source_screen_name'
# on = 'source_type'
# def fix_language(x):
#     if x == -1.0:
#         return 1
#     elif x == 3.0:
#         return 2
#     elif x == 10.0:
#         return 3
#     elif x == 17.0:
#         return 4
#     elif x == 24.0:
#         return 5
#     elif x == 31.0:
#         return 6
#     elif x == 38.0:
#         return 7
#     elif x == 45.0:
#         return 8
#     elif x == 52.0:
#         return 9
#     elif x == 59.0:
#         return 10
#     else:
#         return 6
#
#
# def length_range(x):
#     n = 5000
#     split = np.linspace(185, 12174000, n)
#     its = 1
#     while True:
#         if its == n:
#             return 0
#         else:
#             if x < int(split[its]):
#                 return its
#             else:
#                 its += 1
#
#
# def length_bin_range(x):
#     if x < 500000:
#         return 1
#     else:
#         return 0
#
#
# df['language'] = df['language'].apply(fix_language).astype(np.int8)
# df.song_length.fillna(248770, inplace=True)
# df['song_length'] = df['song_length'].astype(np.int64)
# df['length_range'] = df['song_length'].apply(length_range).astype(np.int64)
# df['length_bin_range'] = df['song_length'].apply(length_bin_range).astype(np.int64)
# df['genre_ids'] = df['genre_ids'].astype(object)
# df['genre_ids'].fillna('no_genre_id', inplace=True)
# df['composer'] = df['composer'].astype(object)
# df['composer'].fillna('no_composer', inplace=True)
# df = df_songs
# df = df_custom_song_data
# df = df_songs_extra
# on = 'song_length'
on = 'genre_ids'
# on = 'artist_name'
# on = 'composer'
# on = 'composer'
# on = 'language'
# on = 'name'
# on = 'isrc'
# on = 'song_year'

# df = df_members
# def get_count(x):
#     try:
#         return count[x]
#     except KeyError:
#         return 0
#
#
# count = pickle.load(open(save_dir + 'song_count_dict.save', "rb"))
# df['song_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + 'liked_song_count_dict.save', "rb"))
# df['liked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + 'disliked_song_count_dict.save', "rb"))
# df['disliked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
#
# count = pickle.load(open(save_dir + 'artist_count_dict.save', "rb"))
# df['artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + 'liked_artist_count_dict.save', "rb"))
# df['liked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + 'disliked_artist_countt_dict.save', "rb"))
# df['disliked_artist_countt'] = df['artist_name'].apply(get_count).astype(np.int64)
# del count

# df['age'] = df['bd'].apply(fix_bd_in_members).astype(np.int8)
# df['age_range'] = df['age'].apply(age_range).astype(np.int8)
# df = df.drop(['bd'], axis=1)
# df['membership_days'] = df['expiration_date'].subtract(df['registration_init_time']).dt.days.astype(int)
# df['membership_days'] = df['membership_days'].apply(no_negative).astype(np.int64)
# df['membership_days_range'] = df['membership_days'].apply(mem_day_range).astype(np.int8)
# df['registration_year'] = df['registration_init_time'].dt.year
# df['registration_month'] = df['registration_init_time'].dt.month
# df['registration_date'] = df['registration_init_time'].dt.day
# df.rename(columns={'expiration_date': 'expiration_time'}, inplace=True)
# df['expiration_year'] = df['expiration_time'].dt.year
# df['expiration_month'] = df['expiration_time'].dt.month
# df['expiration_date'] = df['expiration_time'].dt.day
# # df['gender'].cat.add_categories(['nnn']).fillna('nnn', inplace=True)

# df['sex'] = df['gender'].apply(gender_fix).astype(np.int8)
# df['sex_guess'] = df['sex'].apply(sex_guess).astype(np.int8)
# df = df.drop(['gender'], axis=1)
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
# # print(s)
print('<'*20)
ddd = df.dtypes.to_dict()
for i in ddd:
    on = i
    print('inspecting:', on)
    print('>' * 20)
    print('any null:', df[on].isnull().values.any())
    print('null number:', df[on].isnull().values.sum())
    print('<'*20)
    print()
# # df['membership_days'] = df['expiration_date'].subtract(df['registration_init_time']).dt.days.astype(int)
# # on = 'membership_days'
#
# # dff = pd.DataFrame()
# # dff['gender'] = df['gender'].dropna()
# # print(dff.head())
# # print(len(dff))
# plot = True
plot = False
if plot:
    plt.figure(figsize=(15, 12))
    # dff = pd.DataFrame()
    # dff[on] = df[on].dropna()
    sns.distplot(df[on])
    # sns.countplot(df[on])
    # plt.xlim((0, 0.3))
    plt.show()
    # dff = df.query(on, " >=13 and ", on, " <70")
    # dff = df.query(on+" <0")
    # print(dff[on].describe())
    # plt.figure(figsize=(15, 12))
    # sns.countplot(dff[on])
    # plt.show()
# print(df[df.song_id == missing])
# print(df[df.language.isnull()])
# print(len(df[df.artist_name == 'JONGHYUN']))
# print(df[df.artist_name == 'JONGHYUN'].head())


# def gender_fix(x):
#     if x == 'female':
#         return 1
#     elif x == 'male':
#         return 2
#     else:
#         return 0
# # print(df.head())
#
#
# def gender_imagine(x):
#     if x == 'female':
#         return 1
#     elif x == 'male':
#         return 2
#     else:
#         return np.random.randint(1, 3)

# df['gender'].cat.add_categories(['unknown']).fillna('unknown', inplace=True)
# df['sex_guess'] = df['gender'].apply(gender_fix).astype(np.int8)
'''
def ex_days(x):
    # print(type(x))
    # print(x<200)
    lower = 14
    upper = 60
    if x < lower or x > upper:
        a = 3.5*np.random.normal()+28
        if a < lower:
            return lower
        elif a > upper:
            return upper
        else:
            return a
    else:
        return x


def mem_day_range(x):
    if x < 0:
        return 0
    elif x < 1000:
        return 1
    elif x < 2000:
        return 2
    elif x < 3000:
        return 3
    elif x < 4000:
        return 4
    else:
        return 5
# print(df.head())
'''
count = 0
kinds = set()
# print(df.iloc[[419838]])
# isit = True
isit = False
if isit:

    def isrc_to_year(isrc):
        if type(isrc) == str:
            if int(isrc[5:7]) > 17:
                return 1900 + int(isrc[5:7])
            else:
                return 2000 + int(isrc[5:7])
        else:
            # print('here')
            a = np.random.poisson(2016, 500)
            a = int(np.mean(a))
            if a > 2016:
                a = int(np.random.uniform(1918, 2017))
            return a


    def isrc_to_c(isrc):
        if type(isrc) == str:
            return isrc[0:2]
        else:
            return 'US'

    def isrc_to_rc(isrc):
        if type(isrc) == str:
            return isrc[2:5]
        else:
            return 'no_rc'

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
        global kinds
        if x == 'no_genre_id':
            return 0
        else:
            a = x.count('|') + 1
            if a == 1:
                kinds.add(x)
            else:
                for i in x.split('|'):
                    kinds.add(i)
            return a


    # df['genre_ids_count'] = df['genre_ids'].apply(genre_id_count).astype(np.int8)


    # df['lyricists_count'] = df['lyricist'].apply(lyricist_count).astype(np.int8)
    # df['composer_count'] = df['composer'].apply(composer_count).astype(np.int8)


    # df['song_year'] = df['isrc'].apply(isrc_to_year).astype(np.int64)
    # df['song_year_bin_range'] = df['song_year'].apply(song_year_bin_range).astype(np.int64)
    # df['song_year_chunk_range'] = df['song_year'].apply(song_year_chunk_range).astype(np.int64)
    # df['song_country'] = df['isrc'].apply(isrc_to_c).astype(object)
    # df['rc'] = df['isrc'].apply(isrc_to_rc).astype(object)


    def length_chunk_range(x):

        n = 200
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


    def length_bin_range(x):
        if x < 500000:
            return 1
        else:
            return 0

    # pickle.dump(song_count, open(save_dir+'song_count_dict.save', "wb"))
    # pickle.dump(song_count, open(save_dir+'liked_song_count_dict.save', "wb"))
    # pickle.dump(song_count, open(save_dir + 'disliked_song_count_dict.save', "wb"))
    # pickle.dump(artist_count, open(save_dir+'artist_count_dict.save', "wb"))
    # pickle.dump(artist_count, open(save_dir+'liked_artist_count_dict.save', "wb"))
    # pickle.dump(artist_count, open(save_dir + 'disliked_artist_countt_dict.save', "wb"))



    def get_count(x):
        try:
            return count[x]
        except KeyError:
            return 0


    count = pickle.load(open(save_dir + 'song_count_dict.save', "rb"))
    df['song_count'] = df['song_id'].apply(get_count).astype(np.int64)
    count = pickle.load(open(save_dir + 'liked_song_count_dict.save', "rb"))
    df['liked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
    count = pickle.load(open(save_dir + 'disliked_song_count_dict.save', "rb"))
    df['disliked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)

    count = pickle.load(open(save_dir + 'artist_count_dict.save', "rb"))
    df['artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
    count = pickle.load(open(save_dir + 'liked_artist_count_dict.save', "rb"))
    df['liked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
    count = pickle.load(open(save_dir + 'disliked_artist_count_dict.save', "rb"))
    df['disliked_artist_countt'] = df['artist_name'].apply(get_count).astype(np.int64)
    del count
    # df['artist_composer'] = (df['artist_name'] == df['composer']).astype(np.int8)
    # df['artist_composer_lyricist'] = ((df['artist_name'] == df['composer']) &
    #                                   (df['artist_name'] == df['lyricist']) &
    #                                   (df['composer'] == df['lyricist'])).astype(np.int8)
    # df['song_length'] = df['song_length'].astype(np.int64)
    # df['length_chunk_range'] = df['song_length'].apply(length_chunk_range).astype(np.int64)
    # df['length_bin_range'] = df['song_length'].apply(length_bin_range).astype(np.int64)
    # df['length_range'] = df['song_length'].apply(length_range).astype(np.int64)
    # df['length_bin_range'] = df['song_length'].apply(length_bin_range).astype(np.int64)
    # df['membership_days_range'] = df[on].apply(mem_day_range).astype(np.int8)
    new_on = 'count_song_played'
    # df['age_range'] = df['age'].apply(age_range).astype(np.int8)
    # df = df.drop(['bd'], axis=1)
    # df['age'] = df['bd'].apply(tt)
    # dff = df['age'].dropna()

    # print(len(dff))
    # print(dff.describe())
    # print('after head')
    # print(df.head())
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
    # print('describe: ', df[new_on].describe())
    # c = df.loc[df[new_on] == 1]
    # print(len(c)/len(li))
    # print('kinds len:', len(kinds))
    # print(kinds)
    plot = True
    # plot = False
    if plot:
        plt.figure(figsize=(15, 12))
        # dff = pd.DataFrame()
        # dff[new_on] = df[new_on].dropna()
        # del df
        # sns.distplot(df[new_on])
        sns.countplot(df[new_on])
        # df[on].value_counts(sort=False).plot.bar()
        # plt.xlim([-10, 100])
        plt.show()

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

