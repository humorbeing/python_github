import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()

print('in train, test join and count program.')
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
train = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
load_name = 'test_merge'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
test = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

train.drop('target', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

df = pd.concat([train, test])


# print('creating train set.')
# save_name = 'train_'
# vers = 'set'
# d = df.dtypes.to_dict()
# # print(d)
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)


def IN_TRAIN_Counter(on_in):
    counter = {k: v for k, v in df[on_in].value_counts().iteritems()}
    pickle.dump(counter, open(save_dir+'counter/'+'ITC_'+on_in+'_dict.save', "wb"))
    del counter


cols = df.columns

for col in cols:
    IN_TRAIN_Counter(col)

del df


load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt


def Counter_1111(on_in):
    counter = {k: v for k, v in df[on_in].value_counts().iteritems()}
    pickle.dump(counter, open(save_dir + 'counter/' + 'CC11_' + on_in + '_dict.save', "wb"))
    del counter


df = df[df['target'] == 1]
for col in cols:
    Counter_1111(col)

del df


load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt


counter = {}


def get_count(x):
    try:
        return counter[x]
    except KeyError:
        return 0


def add_this_counter_column(on_in):
    global counter, df
    counter = pickle.load(open(save_dir+'counter/'+'ITC_'+on_in+'_dict.save', "rb"))
    df['ITC_'+on_in] = df[on_in].apply(get_count).astype(np.int64)
    counter = pickle.load(open(save_dir + 'counter/' + 'CC11_' + on_in + '_dict.save', "rb"))
    df['CC11_' + on_in] = df[on_in].apply(get_count).astype(np.int64)
    df.drop(on_in, axis=1, inplace=True)


for col in cols:
    add_this_counter_column(col)


def log10me(x):
    return np.log10(x)


for col in cols:
    colc = 'ITC_'+col
    df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
    col1 = 'CC11_'+col
    df['OinC_'+col] = df[col1]/df[colc]

df = df[[
    'ITC_song_id',
    'CC11_song_id',
    'OinC_song_id',
    'ITC_song_id_log10'
]]
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
print(df.head(20))
# df.to_csv(save_dir+save_name+vers+'.csv', index=False)
# pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))
#
# print('done.')
#
#
# load_name = 'test_merge'
# dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
# df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
# del dt
#
#
# def fix_source_screen_name_fix(x):
#     if x == 'People local' or x == 'Self profile more':
#         return 'Unknown'
#     else:
#         return x
#
#
# df['source_system_tab'] = df['source_system_tab'].astype(object)
# df['source_system_tab'].fillna('null', inplace=True)
# df['source_system_tab_guess'] = df['source_system_tab'].apply(fix_source_system_tab).astype(object)
#
# df['source_screen_name'] = df['source_screen_name'].astype(object)
# df['source_screen_name'].fillna('Unknown', inplace=True)
# df['source_screen_name'] = df['source_screen_name'].apply(fix_source_screen_name_fix).astype(object)
# df['source_screen_name_guess'] = df['source_screen_name'].apply(fix_source_screen_name).astype(object)
#
#
# df['source_type'] = df['source_type'].astype(object)
# df['source_type'].fillna('unknown', inplace=True)
# df['source_type_guess'] = df['source_type'].apply(fix_source_type).astype(object)
#
# df.drop(['source_system_tab_guess',
#          'source_screen_name_guess',
#          'source_type_guess',
#          ],
#         axis=1, inplace=True)
#
# print('creating test set.')
# save_name = 'test_'
# vers = 'set'
# d = df.dtypes.to_dict()
# # print(d)
# print('dtypes of df:')
# print('>'*20)
# print(df.dtypes)
# print('number of columns:', len(df.columns))
# print('number of data:', len(df))
# print('<'*20)
# df.to_csv(save_dir+save_name+vers+'.csv', index=False)
# pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))
#
# print('p4 fixup train, test done.')


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

