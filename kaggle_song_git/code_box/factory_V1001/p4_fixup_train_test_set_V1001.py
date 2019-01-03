import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_merge'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt


def fix_source_system_tab(x):
    if x == 'null':
        return 'my library'
    else:
        return x


def fix_source_screen_name(x):
    if x == 'Unknown':
        return 'Local playlist more'
    else:
        return x


def fix_source_type(x):
    if x == 'unknown':
        return 'local-library'
    else:
        return x


# df['time'] = df.index
df['source_system_tab'] = df['source_system_tab'].astype(object)
df['source_system_tab'].fillna('null', inplace=True)
df['source_system_tab_guess'] = df['source_system_tab'].apply(fix_source_system_tab).astype(object)

df['source_screen_name'] = df['source_screen_name'].astype(object)
df['source_screen_name'].fillna('Unknown', inplace=True)
df['source_screen_name_guess'] = df['source_screen_name'].apply(fix_source_screen_name).astype(object)

df['source_type'] = df['source_type'].astype(object)
df['source_type'].fillna('unknown', inplace=True)
df['source_type_guess'] = df['source_type'].apply(fix_source_type).astype(object)

df.drop(['source_system_tab_guess',
         'source_screen_name_guess',
         'source_type_guess',
         ],
        axis=1, inplace=True)
print('creating train set.')
save_name = 'train_'
vers = 'set'
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

print('done.')


load_name = 'test_merge'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt


def fix_source_screen_name_fix(x):
    if x == 'People local' or x == 'Self profile more':
        return 'Unknown'
    else:
        return x


df['source_system_tab'] = df['source_system_tab'].astype(object)
df['source_system_tab'].fillna('null', inplace=True)
df['source_system_tab_guess'] = df['source_system_tab'].apply(fix_source_system_tab).astype(object)

df['source_screen_name'] = df['source_screen_name'].astype(object)
df['source_screen_name'].fillna('Unknown', inplace=True)
df['source_screen_name'] = df['source_screen_name'].apply(fix_source_screen_name_fix).astype(object)
df['source_screen_name_guess'] = df['source_screen_name'].apply(fix_source_screen_name).astype(object)


df['source_type'] = df['source_type'].astype(object)
df['source_type'].fillna('unknown', inplace=True)
df['source_type_guess'] = df['source_type'].apply(fix_source_type).astype(object)

df.drop(['source_system_tab_guess',
         'source_screen_name_guess',
         'source_type_guess',
         ],
        axis=1, inplace=True)

print('creating test set.')
save_name = 'test_'
vers = 'set'
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

print('p4 fixup train, test done.')


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

