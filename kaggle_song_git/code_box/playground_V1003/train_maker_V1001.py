import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_merge_fixing'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

df['source_system_tab'] = df['source_system_tab'].astype(object)
df['source_system_tab'].fillna('my library', inplace=True)
df['source_screen_name'] = df['source_screen_name'].astype(object)
df['source_screen_name'].fillna('Local playlist more', inplace=True)
df['source_type'] = df['source_type'].astype(object)
df['source_type'].fillna('local-library', inplace=True)

print('creating train set.')
save_name = 'train_'
vers = 'fillna3'
d = df.dtypes.to_dict()
print(d)
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)
pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

print('done.')


plt.show()
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

