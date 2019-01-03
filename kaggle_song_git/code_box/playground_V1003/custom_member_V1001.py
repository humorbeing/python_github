import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np

since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
# df_train = pd.read_csv(data_dir+"train.csv")
#
# df_songs = pd.read_csv(data_dir+"songs.csv")
# #
# df_songs_extra = pd.read_csv(data_dir+"song_extra_info.csv")

df = pd.read_csv(data_dir+"members.csv",
                         dtype={'city': np.uint8,
                                'bd': np.uint8,
                                'gender': 'category',
                                'registered_via': np.uint8,
                                'msno': 'category'
                                },
                         parse_dates=['registration_init_time',
                                      'expiration_date']
                         )

# df_test = pd.read_csv(data_dir+"test.csv")

# df_song_data = pd.read_csv(save_dir+'song_data.csv')

# df_custom_song_data = pd.read_csv(save_dir+'custom_song_data.csv')
# df_custom_members = pd.read_csv(save_dir+'custom_members.csv')


def checking(df1, df2, key, name1='', name2=''):
    set1 = set(df1[key])
    set2 = set(df2[key])
    intersection = set.intersection(set1, set2)
    non_set1 = set.symmetric_difference(set1, intersection)

    print('- '*20)
    print('['+name1+'] and ['+name2+']: keyword['+key+'}')
    print('Number of kinds of ['+key+'] in ['+name1+']: ',
        len(set1))
    print('Number of kinds of [' + key + '] in [' + name2 + ']: ',
        len(set2))
    print('Number of kinds of ['+key+'] in intersection of ['+name1+'] and ['+name2+']: ',
        len(intersection))
    print('Number of kinds of ['+key+'] in ['+name1+'] not in intersection of ['+name1+'] and ['+name2+']: ',
        len(non_set1))
    print()
    return non_set1


def set_in_df(s, df, key, name1='', name2=''):
    set1 = s
    set2 = set(df[key])
    intersection = set.intersection(set1, set2)
    non_set1 = set.symmetric_difference(set1, intersection)
    print('- ' * 20)
    print('[' + name1 + '] and [' + name2 + ']: keyword[' + key + '}')
    print('Number of kinds of [' + key + '] in [' + name1 + ']: ',
          len(set1))
    print('Number of kinds of [' + key + '] in [' + name2 + ']: ',
          len(set2))
    print('Number of kinds of [' + key + '] in intersection of [' + name1 + '] and [' + name2 + ']: ',
          len(intersection))
    print(
        'Number of kinds of [' + key + '] in [' + name1 + '] not in intersection of [' + name1 + '] and [' + name2 + ']: ',
        len(non_set1))
    print()
    return non_set1


def fix_bd_in_members(x):
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


def age_range(x):
    if x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    else:
        return 5


def no_negative(x):
    if x < 0.0:
        return 0
    else:
        return int(x)


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


#
def gender_fix(x):
    if x == 'female':
        return 1
    elif x == 'male':
        return 2
    else:
        # print('hh')
        return 0


#

def sex_guess(x):
    if x == 1:
        return 1
    elif x == 2:
        return 2
    else:
        return np.random.randint(1, 3)

# df = df_train
# df = df_test

# on = 'target'
# on = 'source_system_tab'
# on = 'source_screen_name'
# on = 'source_type'

# df = df_songs
# df = df_custom_song_data
# df = df_songs_extra
# on = 'song_length'
# on = 'genre_ids'
# on = 'artist_name'
# on = 'composer'
# on = 'lyricist'
# on = 'language'
# on = 'name'
# on = 'isrc'

# df = df_members


df['age'] = df['bd'].apply(fix_bd_in_members).astype(np.int8)
df['age_range'] = df['age'].apply(age_range).astype(np.int8)
df = df.drop(['bd'], axis=1)
df['membership_days'] = df['expiration_date'].subtract(df['registration_init_time']).dt.days.astype(int)
df['membership_days'] = df['membership_days'].apply(no_negative).astype(np.int64)
df['membership_days_range'] = df['membership_days'].apply(mem_day_range).astype(np.int8)
df['registration_year'] = df['registration_init_time'].dt.year
df['registration_month'] = df['registration_init_time'].dt.month
df['registration_date'] = df['registration_init_time'].dt.day
df.rename(columns={'expiration_date': 'expiration_time'}, inplace=True)
df['expiration_year'] = df['expiration_time'].dt.year
df['expiration_month'] = df['expiration_time'].dt.month
df['expiration_date'] = df['expiration_time'].dt.day
# df['gender'].cat.add_categories(['nnn']).fillna('nnn', inplace=True)

df['sex'] = df['gender'].apply(gender_fix).astype(np.int8)
df['sex_guess'] = df['sex'].apply(sex_guess).astype(np.int8)
df = df.drop(['gender'], axis=1)
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('<'*20)
on = 'msno'
on = 'city'
# on = 'bd'
# on = 'membership_days'
# on = 'age'
# on = 'age_range'
# on = 'sex'
# on = 'sex_guess'
on = 'registered_via'
on = 'registration_init_time'
on = 'expiration_time'
print('inspecting:', on)
print('>'*20)
print('any null:', df[on].isnull().values.any())
print('null number:', df[on].isnull().values.sum())
# print(df[on].value_counts())
print(on, 'dtype:', df[on].dtypes)
print('describing', on, ':')
print(df[on].describe())
print('<'*20)
l = df[on]
s = set(l)
# print(len(list_train))
print('list len:', len(l))
print('set len:', len(s))
# print(s)
print('<'*20)
# df['membership_days'] = df['expiration_date'].subtract(df['registration_init_time']).dt.days.astype(int)
# on = 'membership_days'

# dff = pd.DataFrame()
# dff['gender'] = df['gender'].dropna()
# print(dff.head())
# print(len(dff))
# plot = True
plot = False
if plot:
    plt.figure(figsize=(15, 12))
    sns.countplot(df[on])
    # # df[on].value_counts(sort=False).plot.bar()
    # # plt.xlim([-10, 100])
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


# df['membership_days_range'] = df[on].apply(mem_day_range).astype(np.int8)
new_on = 'membership_days_range'
# df['age_range'] = df['age'].apply(age_range).astype(np.int8)
# df = df.drop(['bd'], axis=1)
# df['age'] = df['bd'].apply(tt)
# dff = df['age'].dropna()

# print(len(dff))
# print(dff.describe())
# print('after head')
# print(df.head())
li = df[new_on]
s = set(li)
print('after, df len:', len(df))
print('list len:', len(li))
print('set len:', len(s))
plot = True
# plot = False
if plot:
    plt.figure(figsize=(15, 12))
    sns.countplot(df[new_on])
    # df[on].value_counts(sort=False).plot.bar()
    # plt.xlim([-10, 100])
    plt.show()
'''
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

