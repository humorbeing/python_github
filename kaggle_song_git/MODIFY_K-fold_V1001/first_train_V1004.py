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


num_boost_round = 400
early_stopping_rounds = 0
verbose_eval = 10


params_dart = {
    'boosting': 'dart',

    'learning_rate': 0.3,
    'num_leaves': 511,
    'max_depth': -1,

    'lambda_l1': 0,
    'lambda_l2': 0,
    'max_bin': 127,

    'bagging_fraction': 0.8,
    'bagging_freq': 4,
    'bagging_seed': 1,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 1,
}

params_gbdt = {
    'boosting': 'gbdt',

    'learning_rate': 0.02,
    'num_leaves': 511,
    'max_depth': -1,

    'lambda_l1': 0.2,
    'lambda_l2': 0,
    'max_bin': 255,

    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'bagging_seed': 2,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 2,
}
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
        # dfs = divide_df(dfs, 4)
        df1, df3 = fake_df(dfs, size=0.66)
        df1, df2 = fake_df(dfs1, size=0.5)
        dfs = []
        dfs.append(df1)
        dfs.append(df2)
        dfs.append(df3)
        val = divide_df(val, 4)
        show_df(dfs[0])



        model, cols = val_df(
            params_dart, dfs[0], val[0],
            num_boost_round=num_boost_round,
            # early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )


        output = model.predict(dfs[1][cols])
        dfs[1]['from_model1'] = output
        output = model.predict(dfs[2][cols])
        dfs[2]['from_model1'] = output
        output = model.predict(dfs[3][cols])
        dfs[3]['from_model1'] = output
        output = model.predict(val[1][cols])
        val[1]['from_model1'] = output
        # output = model.predict(val[2][cols])
        # val[2]['from_model1'] = output
        # output = model.predict(val[3][cols])
        # val[3]['from_model1'] = output
        del model

        model, cols = val_df(
            params_gbdt, dfs[1], val[1],
            num_boost_round=num_boost_round,
            # early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

        output = model.predict(dfs[2][cols])
        dfs[2]['from_model2'] = output
        output = model.predict(dfs[3][cols])
        dfs[3]['from_model2'] = output
        output = model.predict(val[2][cols])
        val[2]['from_model2'] = output
        output = model.predict(val[3][cols])
        val[3]['from_model2'] = output

        model, cols = val_df(
            params, dfs[2], val[2],
            num_boost_round=num_boost_round,
            # early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        output = model.predict(dfs[3][cols])
        dfs[3]['from_model3'] = output
        output = model.predict(val[3][cols])
        val[3]['from_model3'] = output
        show_df(val[3])


        model, cols = val_df(
            params1, dfs[3], val[3],
            num_boost_round=num_boost_round,
            # early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )


import operator
sorted_x = sorted(result.items(), key=operator.itemgetter(1))

for i in sorted_x:
    name = i[0] + ':  '
    name = name.rjust(40)
    name = name + str(i[1])
    print(name)