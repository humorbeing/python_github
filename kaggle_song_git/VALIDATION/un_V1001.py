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
load_name = 'train_set.csv'

df1 = read_df(load_name, read_from='../saves01/')
on = [
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'target',
    # 'expiration_month',
    'genre_ids',
    'artist_name',
    'composer',
    'lyricist',
    'language',
    # 'name',
    'song_year',
    'song_country',
    'rc',
    # 'isrc_rest',
    'top1_in_song',
    'top2_in_song',
    'top3_in_song',
]
df1 = df1[on]

df2 = read_df(load_name, read_from='../saves02/')

on = [
    # 'target',
        'membership_days',
    # 'bd_log10',
    # 'expiration_month_log10',
    # 'IMC_expiration_month_log10',
    # 'bd_fixed_log10',
    # 'age_guess_log10',
    # 'bd_range_log10',
    # 'age_guess_range_log10',
    # 'bd_fixed_range_log10',
    # 'IMC_bd_log10',
    # 'IMC_bd_fixed_log10',
    # 'IMC_age_guess_log10',
    # 'IMC_bd_range_log10',
    # 'IMC_bd_fixed_range_log10',
    # 'IMC_age_guess_range_log10',
    # 'IMC_membership_days_log10',
        'song_year',
        # 'ISC_genre_ids',
        'ISC_top1_in_song',
        'ISC_top2_in_song',
        'ISC_top3_in_song',
        # 'ISCZ_artist_name',
        # 'ISC_composer',
        # 'ISCZ_lyricist',
        'ISC_language',
        'ISCZ_rc',
        'ISCZ_isrc_rest',
        'ISC_song_year',
        # 'ISCZ_song_year',
        'song_length_log10',
        'ISCZ_genre_ids_log10',
        'ISC_artist_name_log10',
        'ISCZ_composer_log10',
        'ISC_lyricist_log10',
    # 'ISC_name_log10',
    # 'ISCZ_name_ln',
        'ISC_song_country_ln',
        # 'ISCZ_song_country_log10',
        # 'ISC_rc_ln',
    # 'ISC_isrc_rest_log10',
]
df2 = df2[on]
df2.rename(columns={'song_year': 'song_year_int'}, inplace=True)

df = df1.join(df2)
del df1, df2
show_df(df)

save_me = True
# save_me = False
if save_me:
    save_df(df)







print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))