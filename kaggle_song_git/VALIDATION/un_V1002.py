import sys
sys.path.insert(0, '../')
from me import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pickle

since = time.time()
result = {}
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_me_pure.csv'

df = read_df(load_name)

cols = [
    # 'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    'composer',
    'lyricist',
    # 'language',
    'song_year',
    'top1_in_song',
    'top2_in_song',
    'top3_in_song',
]
df = add_ITC(df, cols)
cols = [
    'msno',
    'language',
]
df = add_11(df, cols)

show_df(df)
save_me = True
# save_me = False
if save_me:
    save_df(df)







print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))