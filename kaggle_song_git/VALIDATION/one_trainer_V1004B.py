import sys
sys.path.insert(0, '../')
from me import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pickle

since = time.time()
data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_me_play.csv'
# df = pd.read_csv('../saves/train_me_play.csv')
# def intme(x):
#     return int(x)
#
df = read_df(load_name)
# df['song_year'] = df['song_year'].astype(object)
# df['song_year_int'] = df['song_year'].apply(intme).astype(np.int64)
# df['song_year'] = df['song_year'].astype('category')
#
# # show_df(df)
# cols = [
#     'msno',
#     'song_id',
#     # 'artist_name',
#     'top1_in_song',
#     # 'top2_in_song',
#     'top3_in_song',
#     # 'language',
#     'song_year',
#     # 'composer',
#     # 'lyricist',
#     'source_screen_name',
#     'source_type',
# ]
# df = add_ITC(df, cols)

show_df(df)


num_boost_round = 500
early_stopping_rounds = 50
verbose_eval = 10

boosting = 'gbdt'

learning_rate = 0.032
num_leaves = 750
max_depth = 50

max_bin = 172
lambda_l1 = 0.2
lambda_l2 = 0


bagging_fraction = 0.9
bagging_freq = 2
bagging_seed = 2
feature_fraction = 0.9
feature_fraction_seed = 2

params = {
    'boosting': boosting,

    'learning_rate': learning_rate,
    'num_leaves': num_leaves,
    'max_depth': max_depth,

    'lambda_l1': lambda_l1,
    'lambda_l2': lambda_l2,
    'max_bin': max_bin,

    'bagging_fraction': bagging_fraction,
    'bagging_freq': bagging_freq,
    'bagging_seed': bagging_seed,
    'feature_fraction': feature_fraction,
    'feature_fraction_seed': feature_fraction_seed,
}
fixed = [
    'target',
    'msno',
    'song_id',
    # 'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    'song_year',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    'top2_in_song',
    # 'top3_in_song',
    # 'rc',

    # 'ITC_source_system_tab_log10_1',
    # 'ITC_source_screen_name_log10_1',
    # 'ITC_source_type_log10_1',
    # 'ITC_artist_name_log10_1',
    # 'FAKE_1512883008',
]

result = {}

for w in df.columns:
    print("'{}',".format(w))

work_on = [
    # 'ITC_msno',
    # 'CC11_msno',
    # 'ITC_name',
    # 'language',

    # 'CC11_name',
    'song_year_int',
    'ITC_song_year_log10_1',
    'ITC_source_screen_name_log10_1',
    'ITC_source_type_log10_1',
    # 'ITC_language_log10_1',
    'ITC_top1_in_song_log10_1',
    # 'ITC_top2_in_song_log10_1',
    'ITC_top3_in_song_log10_1',
    # 'ITC_composer_log10_1',
    # 'ITC_lyricist_log10_1',
    # 'ITC_artist_name_log10_1',

    # 'ITC_song_id_log10_1',
    # 'ITC_song_id_x_1',
    # 'OinC_song_id',
    # 'ITC_msno_log10',
    # 'ITC_msno_log10_1',
    # 'ITC_msno_x_1',
    # 'OinC_msno',
    # 'ITC_name_log10',
    # 'ITC_name_log10_1',
    # 'ITC_name_x_1',
    # 'OinC_name',
]
for w in df.columns:
# for w in work_on:
    if w in fixed:
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
        df_on = df[toto]
        show_df(df_on)
        # save_me = True
        save_me = False
        if save_me:
            save_df(df_on)

        train, val = fake_df(df_on)
        del df_on
        model, cols = val_df(
            params, train, val,
            num_boost_round,
            early_stopping_rounds,
            verbose_eval
        )
        del train, val
        print('complete on:', w)
        result[w] = show_mo(model)


import operator
sorted_x = sorted(result.items(), key=operator.itemgetter(1))
for i in sorted_x:
    name = i[0] + ':  '
    name = name.rjust(40)
    name = name + str(i[1])
    print(name)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))