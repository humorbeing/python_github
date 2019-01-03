import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
df_train = pd.read_csv(data_dir+"train.csv")

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
df_test = pd.read_csv(data_dir+"test.csv")


print('creating custom member.')

set1 = set(df_train['msno'])
set2 = set(df_test['msno'])
union_member = set.union(set1, set2)
# print(len(union_member))
df.set_index('msno', inplace=True)
df = df.loc[union_member]
df.to_csv(save_dir+'custom_members.csv')
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('<'*20)
# print(d)
# pickle.dump(d, open(save_dir+"custom_member_dict.save", "wb"))
# xxxx = pickle.load(open("xxx.save", "rb"))
print('done.')
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

