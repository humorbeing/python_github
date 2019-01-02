import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle
from sklearn.model_selection import train_test_split
from me import *

since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
load_name = 'custom_members_fixed.csv'
load_name = 'custom_song_fixed.csv'
load_name = 'train_set.csv'
df = read_df(load_name)
show_df(df)

cols = ['song_id', 'msno']

df = add_ITC(df, cols)

show_df(df)


num_boost_round = 5000
early_stopping_rounds = 20
verbose_eval = 10

params = []

param1 = {
    'boosting': 'gbdt',

    'learning_rate': 0.5,
    'num_leaves': 15,
    'max_depth': 5,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 255,

    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}
param2 = {
    'boosting': 'gbdt',

    'learning_rate': 0.3,
    'num_leaves': 31,
    'max_depth': 6,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 255,


    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}
param3 = {
    'boosting': 'gbdt',

    'learning_rate': 0.1,
    'num_leaves': 511,
    'max_depth': 10,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 255,

    'you can set min_data': 1,
    'min_data_in_bin': 1,

    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}
param4 = {
    'boosting': 'gbdt',

    'learning_rate': 0.02,
    'num_leaves': 511,
    'max_depth': -1,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 255,

    'you can set min_data': 1,
    'min_data_in_bin': 1,

    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}
params.append(param1)
params.append(param2)
params.append(param3)
params.append(param4)
# on = [
#     'msno',
#     'song_id',
#     'target',
#     'source_system_tab',
#     'source_screen_name',
#     'source_type',
#     'language',
#     'artist_name',
#     'song_count',
#     'member_count',
#     'song_year',
# ]
# df = df[on]
fixed = [
    'target',
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    'song_year',
    # 'language',
    'top3_in_song',
    'ITC_song_id_log10_1',
    # 'ITC_msno_log10_1',
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
    # 'CC11_name',
    # 'ITC_song_id_log10',
    # 'ITC_song_id_log10_1',
    # 'ITC_song_id_x_1',
    # 'OinC_song_id',
    # 'ITC_msno_log10',
    'ITC_msno_log10_1',
    # 'ITC_msno_x_1',
    # 'OinC_msno',
    # 'ITC_name_log10',
    # 'ITC_name_log10_1',
    # 'ITC_name_x_1',
    # 'OinC_name',
]


for w in work_on:
    if w in fixed:
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
        df_on = df[toto]

        # save_me = True
        save_me = False
        if save_me:
            save_df(df_on)

        dfs, val = fake_df(df_on)
        dfs = divide_df(dfs, 4)

        show_df(dfs[0])

        for o in range(4):

            model, cols = val_df(
                params[o], dfs[o], val,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
            )

            for i in range(o+1, 4):
                # print(i)
                dfs[i] = add_column(model, cols, dfs[i], 'from_model'+str(o))

            val = add_column(model, cols, val, 'from_model'+str(o))





print('done')