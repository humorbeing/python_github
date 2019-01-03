import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'

df = pd.read_csv(data_dir+"members.csv",
                 dtype={'city': 'category',
                        'bd': np.uint64,
                        'gender': 'category',
                        'registered_via': 'category',
                        'msno': 'category'
                        },
                 parse_dates=['registration_init_time',
                              'expiration_date']
                 )

df = df[[
    'msno',
    'city',
    'gender',
    'registered_via',
    'registration_init_time',
    'expiration_date',
]]
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

# df['age'] = df['bd'].apply(fix_bd_in_members).astype(np.int8)
# df['age_range'] = df['age'].apply(age_range).astype(np.int8)
# df = df.drop(['bd'], axis=1)

# df['membership_days'] = df['expiration_date'].subtract(df['registration_init_time']).dt.days.astype(int)
# df['membership_days'] = df['membership_days'].apply(no_negative).astype(np.int64)
# df['membership_days_range'] = df['membership_days'].apply(mem_day_range).astype(np.int8)
df['gender'] = df['gender'].astype(object)
df['gender'].fillna('MISSING', inplace=True)
# df.rename(columns={'expiration_date': 'expiration_time'}, inplace=True)
# df['expiration_year'] = df['expiration_time'].dt.year
# df['expiration_month'] = df['expiration_time'].dt.month

# df['registration_year'] = df['registration_init_time'].dt.year
# df['registration_month'] = df['registration_init_time'].dt.month
# df['registration_date'] = df['registration_init_time'].dt.day
# df.rename(columns={'expiration_date': 'expiration_time'}, inplace=True)
# df['expiration_year'] = df['expiration_time'].dt.year
# df['expiration_month'] = df['expiration_time'].dt.month
# df['expiration_date'] = df['expiration_time'].dt.day
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


# df['registration_year'] = df['registration_init_time'].dt.year
# df['registration_month'] = df['registration_init_time'].dt.month
# df['registration_date'] = df['registration_init_time'].dt.day
# df.rename(columns={'expiration_date': 'expiration_time'}, inplace=True)
# df['expiration_year'] = df['expiration_time'].dt.year
# df['expiration_month'] = df['expiration_time'].dt.month
# df['expiration_date'] = df['expiration_time'].dt.day


# df['registration_year'] = df['registration_year'].astype('category')
# df['registration_month'] = df['registration_month'].astype('category')
# df['registration_date'] = df['registration_date'].astype('category')

# df['expiration_year'] = df['expiration_year'].astype('category')
# df['expiration_month'] = df['expiration_month'].astype('category')
# df['expiration_date'] = df['expiration_date'].astype('category')


# df = df.drop(['gender'], axis=1)
# df = df.drop(['expiration_time'], axis=1)
# df = df.drop(['registration_init_time'], axis=1)
# df['registered_via'] = df['registered_via'].astype('category')
count = {}
# count1 = {}
# count2 = {}


def get_count1(x):
    try:
        return count[x]
    except KeyError:
        return 1


# Fake!!!!!!!!!!!!!!!!!!!!!!!!!
# storage = '../fake/'
# count = pickle.load(open(storage + 'total_member_count_dict.save', "rb"))
# df['fake_member_count'] = df['msno'].apply(get_count1).astype(np.int64)
# # Real!!!!!!!!!!!!!!!!!!!!!!!!!
# count = pickle.load(open(save_dir + 'total_member_count_dict.save', "rb"))
# df['member_count'] = df['msno'].apply(get_count1).astype(np.int64)


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



df.drop([
         'city',
         # 'bd',
         # 'gender',
         'registered_via',
         'registration_init_time',
         'expiration_date'
         ],
        axis=1, inplace=True)

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

