import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
import math

since = time.time()


data_dir = '../data/'
save_dir = '../saves/'

df = pd.read_csv(data_dir+"members.csv",
                 dtype={'city': 'category',
                        'bd': np.int64,
                        'gender': 'category',
                        'registered_via': 'category',
                        'msno': 'category'
                        },
                 parse_dates=['registration_init_time',
                              'expiration_date']
                 )

tellme = False
if tellme:
    print()
    print('>'*20)
    print('>'*20)
    print('dtypes of df:')

    print(df.dtypes)
    print('number of rows:', len(df))
    print('number of columns:', len(df.columns))


    for on in df.columns:
        print()
        print('inspecting:', on)
        # print('>'*20)
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
        print()
    print('<'*20)
    print('<'*20)
    print('<'*20)


def fix_bd(x):
    lower = 14
    upper = 60
    if x < lower or x > upper:
        return 0
    else:
        return x



def fix_bd_in_members(x):
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
    if x < 14:
        return 0
    elif x < 20:
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


def gender_fix(x):
    if x == 'female':
        return 1
    elif x == 'male':
        return 2
    else:
        return 0


def sex_guess(x):
    if x == 1:
        return 1
    elif x == 2:
        return 2
    else:
        return np.random.randint(1, 3)
        # return 2


def sex_fre(x):
    if x == 1:
        return 1
    elif x == 2:
        return 2
    else:
        # return np.random.randint(1, 3)
        return 2


# df['mn'] = df.index

# df['age_guess'] = df['bd'].apply(fix_bd_in_members).astype(np.int64)
# df['bd_range_guess'] = df['bd'].apply(age_range).astype(np.int64)
# df['age_range_guess'] = df['age_guess'].apply(age_range).astype(np.int64)
# df = df.drop(['bd'], axis=1)

df['membership_days'] = df['expiration_date'].subtract(df['registration_init_time']).dt.days.astype(int)
df['membership_days'] = df['membership_days'].apply(no_negative).astype(np.int64)
df['membership_days_range'] = df['membership_days'].apply(mem_day_range).astype(np.int64)


# df['registration_year'] = df['registration_init_time'].dt.year
# df['registration_month'] = df['registration_init_time'].dt.month
# df['registration_date'] = df['registration_init_time'].dt.day
# df.rename(columns={'expiration_date': 'expiration_time'}, inplace=True)
# df['expiration_year'] = df['expiration_time'].dt.year
# df['expiration_month'] = df['expiration_time'].dt.month
# df['expiration_date'] = df['expiration_time'].dt.day

df['gender'] = df['gender'].astype(object)
df['gender'].fillna('MISSING', inplace=True)
# df['sex'] = df['gender'].apply(gender_fix).astype(np.int8)
#
# df['sex_guess1'] = df['sex'].apply(sex_guess).astype(np.int8)
# df['sex_guess2'] = df['sex'].apply(sex_guess).astype(np.int8)
# df['sex_guess3'] = df['sex'].apply(sex_guess).astype(np.int8)
# df['sex_guess4'] = df['sex'].apply(sex_guess).astype(np.int8)
# df['sex_guess5'] = df['sex'].apply(sex_guess).astype(np.int8)
#
# df['sex_freq_member'] = df['sex'].apply(sex_fre).astype(np.int8)
#
# df['sex'] = df['sex'].astype('category')
# df['sex_guess1'] = df['sex_guess1'].astype('category')
# df['sex_guess2'] = df['sex_guess2'].astype('category')
# df['sex_guess3'] = df['sex_guess3'].astype('category')
# df['sex_guess4'] = df['sex_guess4'].astype('category')
# df['sex_guess5'] = df['sex_guess5'].astype('category')
#
# df['sex_freq_member'] = df['sex_freq_member'].astype('category')


df['registration_year'] = df['registration_init_time'].dt.year
df['registration_month'] = df['registration_init_time'].dt.month
df['registration_date'] = df['registration_init_time'].dt.day
df.rename(columns={'expiration_date': 'expiration_time'}, inplace=True)
df['expiration_year'] = df['expiration_time'].dt.year
df['expiration_month'] = df['expiration_time'].dt.month
df['expiration_date'] = df['expiration_time'].dt.day


# df['registration_year'] = df['registration_year'].astype('category')
# df['registration_month'] = df['registration_month'].astype('category')
# df['registration_date'] = df['registration_date'].astype('category')
#
# df['expiration_year'] = df['expiration_year'].astype('category')
# df['expiration_month'] = df['expiration_month'].astype('category')
# df['expiration_date'] = df['expiration_date'].astype('category')


# df = df.drop(['gender'], axis=1)
# df = df.drop(['expiration_time'], axis=1)
# df = df.drop(['registration_init_time'], axis=1)
# df['registered_via'] = df['registered_via'].astype('category')
counter = {'MISSING': 0}
# count1 = {}
# count2 = {}


def get_count(x):
    try:
        return counter[x]
    except KeyError:
        return 1


# Fake!!!!!!!!!!!!!!!!!!!!!!!!!
# storage = '../fake/'
# count = pickle.load(open(storage + 'total_member_count_dict.save', "rb"))
# df['fake_member_count'] = df['msno'].apply(get_count1).astype(np.int64)
# # Real!!!!!!!!!!!!!!!!!!!!!!!!!
# count = pickle.load(open(save_dir + 'total_member_count_dict.save', "rb"))
# df['member_count'] = df['msno'].apply(get_count1).astype(np.int64)


def count_up(on):
    global counter, df
    print()
    print('>'*30)
    print()
    print('working on:', on)
    counter = {k: v for k, v in df[on].value_counts().iteritems()}
    df['IMC_'+on] = df[on].apply(get_count).astype(np.int64)
    if 'MISSING' in counter:
        print('MISSING:', counter['MISSING'])
        counter['MISSING'] = 0
        df['IMCZ_'+on] = df[on].apply(get_count).astype(np.int64)
    print('counter len:', len(counter))
    # for i in counter:
    #     print(i, ':', counter[i])
    del counter
    print()
    print('<' * 30)


count_up('city')
count_up('gender')
count_up('registered_via')

count_up('registration_year')
count_up('registration_month')
count_up('registration_date')

count_up('expiration_year')
count_up('expiration_month')
count_up('expiration_date')


df['bd_fixed'] = df['bd'].apply(fix_bd).astype(np.int64)
df['age_guess'] = df['bd'].apply(fix_bd_in_members).astype(np.int64)
df['bd_range'] = df['bd'].apply(age_range).astype(np.int64)
df['age_guess_range'] = df['age_guess'].apply(age_range).astype(np.int64)
df['bd_fixed_range'] = df['bd_fixed'].apply(age_range).astype(np.int64)
count_up('bd')
count_up('bd_fixed')
count_up('age_guess')
count_up('bd_range')
count_up('bd_fixed_range')
count_up('age_guess_range')

# df['membership_days'] = df['expiration_date'].subtract(df['registration_init_time']).dt.days.astype(int)
# df['membership_days_fixed'] = df['membership_days'].apply(no_negative).astype(np.int64)
# df['membership_days_range'] = df['membership_days_fixed'].apply(mem_day_range).astype(np.int64)

count_up('membership_days')
# count_up('membership_days_fixed')
count_up('membership_days_range')


def log10me(x):
    return np.log10(x)


def log2me(x):
    return np.log2(x)


def lnme(x):
    return np.log(x)


df.drop([
         'city',
         # 'bd',
         'gender',
         'registered_via',
         'registration_init_time',
         'expiration_time'
         ],
        axis=1, inplace=True)

ccc = [i for i in df.columns]
print(type(ccc))
ccc.remove('msno')
for col in df.columns:
    if not col == 'msno':
        df[col+'_log10'] = df[col].apply(log10me).astype(np.float64)
        # df[col + '_log2'] = df[col].apply(log2me).astype(np.float64)
        # df[col + '_ln'] = df[col].apply(lnme).astype(np.float64)
df.drop(ccc, axis=1, inplace=True)

tellme = False
if tellme:
    print()
    print('>'*20)
    print('>'*20)

    for on in df.columns:
        print()
        print('inspecting:', on)
        # print('>'*20)
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
        print()
    print('dtypes of df:')

    print(df.dtypes)
    print('number of rows:', len(df))
    print('number of columns:', len(df.columns))
    print('<'*20)
    print('<'*20)
    print('<'*20)

print('creating custom member.')

print('dtypes of df:')
d = df.dtypes.to_dict()
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+'custom_members_fixed.csv', index=False)

pickle.dump(d, open(save_dir+"custom_members_fixed_dict.save", "wb"))


print('done.')
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

