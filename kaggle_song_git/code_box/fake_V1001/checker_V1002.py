import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_path = '../data/'
save_dir = '../fake/'
if 'fake' in save_dir:
    print('-' * 45)
    print()
    print(' !' * 22)
    print()
    print('  this is fake world  ' * 2)
    print()
    print(' !' * 22)
    print()
    print('-' * 45)
df1 = pd.read_csv(save_dir + 'fake_train.csv',
                dtype={'msno': 'category',
                       'source_system_tab': 'category',
                       'source_screen_name': 'category',
                       'source_type': 'category',
                       'target': np.uint8,
                       'song_id': 'category'
                       }
                )
df2 = pd.read_csv(save_dir + 'fake_test.csv',
                dtype={'msno': 'category',
                       'source_system_tab': 'category',
                       'source_screen_name': 'category',
                       'source_type': 'category',
                       'id': 'category',
                       'song_id': 'category'
                       }
                )

set_df1_msno = set(df1['msno'])
print('set df1 msno len:', len(set_df1_msno))
set_df2_msno = set(df2['msno'])
print('set df2 msno len:', len(set_df2_msno))
inter_set = set.intersection(set_df1_msno, set_df2_msno)
print('inters. len:', len(inter_set))
print('proportion:', len(inter_set)/len(set_df2_msno))
print()
set_df1_msno = set(df1['song_id'])
print('set df1 song_id len:', len(set_df1_msno))
set_df2_msno = set(df2['song_id'])
print('set df2 song_id len:', len(set_df2_msno))
inter_set = set.intersection(set_df1_msno, set_df2_msno)
print('inters. len:', len(inter_set))
print('proportion:', len(inter_set)/len(set_df2_msno))

if 'fake' in save_dir:
    print('-' * 45)
    print()
    print(' !' * 22)
    print()
    print('  this is fake world  ' * 2)
    print()
    print(' !' * 22)
    print()
    print('-' * 45)
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


