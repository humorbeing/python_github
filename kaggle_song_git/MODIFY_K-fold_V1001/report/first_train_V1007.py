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


num_boost_round = 500
early_stopping_rounds = 20
verbose_eval = 10

params = []

param1 = {
    'boosting': 'dart',

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
on1 = ['top3_in_song',
    'ITC_song_id_log10_1',
'msno',
    'song_id',
'target',
]
on2 = ['target',
'artist_name',
    'song_year',
    'msno',]
on3 = ['song_id',
'target',
    'source_system_tab',
    'source_screen_name',]
on4 = ['target',
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type',
    'artist_name',
    'song_year',
    # 'language',
    'top3_in_song',
    'ITC_song_id_log10_1',]
ons = []
ons.append(on1)
ons.append(on2)
ons.append(on3)
ons.append(on4)
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
                params[o], dfs[o][ons[o]], val[ons[o]],
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
            )

            for i in range(o+1, 4):
                # print(i)
                dfs[i] = add_column(model, cols, dfs[i], 'from_model'+str(o))

            val = add_column(model, cols, val, 'from_model'+str(o))





print('done')

'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/MODIFY_K-fold_V1001/first_train_V1007.py

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                  category
song_id               category
source_system_tab     category
source_screen_name    category
source_type           category
target                   uint8
gender                category
artist_name           category
composer              category
lyricist              category
language              category
name                  category
song_year             category
song_country          category
rc                    category
isrc_rest             category
top1_in_song          category
top2_in_song          category
top3_in_song          category
dtype: object
number of rows: 7377418
number of columns: 19
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
target                    uint8
gender                 category
artist_name            category
composer               category
lyricist               category
language               category
name                   category
song_year              category
song_country           category
rc                     category
isrc_rest              category
top1_in_song           category
top2_in_song           category
top3_in_song           category
ITC_song_id               int64
ITC_msno                  int64
ITC_song_id_log10_1     float64
ITC_msno_log10_1        float64
dtype: object
number of rows: 7377418
number of columns: 23
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
'msno',
'song_id',
'source_system_tab',
'source_screen_name',
'source_type',
'target',
'gender',
'artist_name',
'composer',
'lyricist',
'language',
'name',
'song_year',
'song_country',
'rc',
'isrc_rest',
'top1_in_song',
'top2_in_song',
'top3_in_song',
'ITC_song_id',
'ITC_msno',
'ITC_song_id_log10_1',
'ITC_msno_log10_1',
working on: ITC_msno_log10_1

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                    uint8
msno                   category
song_id                category
source_system_tab      category
source_screen_name     category
source_type            category
artist_name            category
song_year              category
top3_in_song           category
ITC_song_id_log10_1     float64
ITC_msno_log10_1        float64
dtype: object
number of rows: 1401710
number of columns: 11
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:671: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.654058	valid_1's auc: 0.598175
[20]	training's auc: 0.674553	valid_1's auc: 0.606068
[30]	training's auc: 0.689291	valid_1's auc: 0.610535
[40]	training's auc: 0.697633	valid_1's auc: 0.613178
[50]	training's auc: 0.702663	valid_1's auc: 0.615279
[60]	training's auc: 0.704216	valid_1's auc: 0.616805
[70]	training's auc: 0.707083	valid_1's auc: 0.617793
[80]	training's auc: 0.708059	valid_1's auc: 0.617975
[90]	training's auc: 0.709277	valid_1's auc: 0.618787
[100]	training's auc: 0.711246	valid_1's auc: 0.619932
[110]	training's auc: 0.712593	valid_1's auc: 0.620299
[120]	training's auc: 0.714584	valid_1's auc: 0.620831
[130]	training's auc: 0.716719	valid_1's auc: 0.621384
[140]	training's auc: 0.717937	valid_1's auc: 0.621894
[150]	training's auc: 0.719148	valid_1's auc: 0.622165
[160]	training's auc: 0.719758	valid_1's auc: 0.622565
[170]	training's auc: 0.7227	valid_1's auc: 0.623517
[180]	training's auc: 0.723828	valid_1's auc: 0.624117
[190]	training's auc: 0.726414	valid_1's auc: 0.624587
[200]	training's auc: 0.728291	valid_1's auc: 0.62498
[210]	training's auc: 0.72842	valid_1's auc: 0.62529
[220]	training's auc: 0.730049	valid_1's auc: 0.625397
[230]	training's auc: 0.731121	valid_1's auc: 0.625754
[240]	training's auc: 0.733268	valid_1's auc: 0.626089
[250]	training's auc: 0.733542	valid_1's auc: 0.626022
[260]	training's auc: 0.734656	valid_1's auc: 0.6261
Early stopping, best iteration is:
[245]	training's auc: 0.732908	valid_1's auc: 0.62636
[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.628558	valid_1's auc: 0.570833
[20]	training's auc: 0.656107	valid_1's auc: 0.581954
[30]	training's auc: 0.674137	valid_1's auc: 0.589359
[40]	training's auc: 0.684797	valid_1's auc: 0.592671
[50]	training's auc: 0.693706	valid_1's auc: 0.59672
[60]	training's auc: 0.699756	valid_1's auc: 0.600118
[70]	training's auc: 0.704439	valid_1's auc: 0.602037
[80]	training's auc: 0.707619	valid_1's auc: 0.603213
[90]	training's auc: 0.711267	valid_1's auc: 0.604723
[100]	training's auc: 0.714713	valid_1's auc: 0.605894
[110]	training's auc: 0.717291	valid_1's auc: 0.606798
[120]	training's auc: 0.719531	valid_1's auc: 0.607472
[130]	training's auc: 0.721747	valid_1's auc: 0.607935
[140]	training's auc: 0.723404	valid_1's auc: 0.60863
[150]	training's auc: 0.725729	valid_1's auc: 0.609316
[160]	training's auc: 0.72697	valid_1's auc: 0.609816
[170]	training's auc: 0.728436	valid_1's auc: 0.610298
[180]	training's auc: 0.729659	valid_1's auc: 0.610462
[190]	training's auc: 0.730699	valid_1's auc: 0.610741
[200]	training's auc: 0.731858	valid_1's auc: 0.611273
[210]	training's auc: 0.73267	valid_1's auc: 0.611401
[220]	training's auc: 0.733574	valid_1's auc: 0.61149
[230]	training's auc: 0.7341	valid_1's auc: 0.611727
[240]	training's auc: 0.734994	valid_1's auc: 0.612016
[250]	training's auc: 0.735478	valid_1's auc: 0.612264
[260]	training's auc: 0.736183	valid_1's auc: 0.612387
[270]	training's auc: 0.73663	valid_1's auc: 0.61237
[280]	training's auc: 0.737063	valid_1's auc: 0.612603
[290]	training's auc: 0.737425	valid_1's auc: 0.612703
[300]	training's auc: 0.737679	valid_1's auc: 0.612938
[310]	training's auc: 0.737939	valid_1's auc: 0.612817
[320]	training's auc: 0.738319	valid_1's auc: 0.612838
Early stopping, best iteration is:
[300]	training's auc: 0.737679	valid_1's auc: 0.612938
[LightGBM] [Warning] Unknown parameter you
[LightGBM] [Warning] Unknown parameter can
[LightGBM] [Warning] Unknown parameter set
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.688299	valid_1's auc: 0.612761
[20]	training's auc: 0.690823	valid_1's auc: 0.614024
[30]	training's auc: 0.692686	valid_1's auc: 0.61472
[40]	training's auc: 0.694413	valid_1's auc: 0.615184
[50]	training's auc: 0.695906	valid_1's auc: 0.61567
[60]	training's auc: 0.69717	valid_1's auc: 0.616057
[70]	training's auc: 0.697971	valid_1's auc: 0.616237
[80]	training's auc: 0.698812	valid_1's auc: 0.616446
[90]	training's auc: 0.699382	valid_1's auc: 0.616589
[100]	training's auc: 0.700115	valid_1's auc: 0.616704
[110]	training's auc: 0.700881	valid_1's auc: 0.616807
[120]	training's auc: 0.701438	valid_1's auc: 0.616924
[130]	training's auc: 0.702035	valid_1's auc: 0.617042
[140]	training's auc: 0.70242	valid_1's auc: 0.617115
[150]	training's auc: 0.702961	valid_1's auc: 0.617159
[160]	training's auc: 0.70343	valid_1's auc: 0.617202
[170]	training's auc: 0.703856	valid_1's auc: 0.617252
[180]	training's auc: 0.704212	valid_1's auc: 0.61727
[190]	training's auc: 0.704571	valid_1's auc: 0.617295
[200]	training's auc: 0.704993	valid_1's auc: 0.617252
Early stopping, best iteration is:
[184]	training's auc: 0.704413	valid_1's auc: 0.617322
[LightGBM] [Warning] Unknown parameter you
[LightGBM] [Warning] Unknown parameter can
[LightGBM] [Warning] Unknown parameter set
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.784161	valid_1's auc: 0.656911
[20]	training's auc: 0.790233	valid_1's auc: 0.659078
[30]	training's auc: 0.796313	valid_1's auc: 0.661291
[40]	training's auc: 0.801609	valid_1's auc: 0.663181
[50]	training's auc: 0.805492	valid_1's auc: 0.664618
[60]	training's auc: 0.809062	valid_1's auc: 0.665472
[70]	training's auc: 0.813816	valid_1's auc: 0.667301
[80]	training's auc: 0.818108	valid_1's auc: 0.668889
[90]	training's auc: 0.821076	valid_1's auc: 0.669744
[100]	training's auc: 0.824861	valid_1's auc: 0.670944
[110]	training's auc: 0.827505	valid_1's auc: 0.671584
[120]	training's auc: 0.830802	valid_1's auc: 0.672755
[130]	training's auc: 0.833792	valid_1's auc: 0.673805
[140]	training's auc: 0.836744	valid_1's auc: 0.674731
[150]	training's auc: 0.839418	valid_1's auc: 0.675546
[160]	training's auc: 0.841571	valid_1's auc: 0.676106
[170]	training's auc: 0.843849	valid_1's auc: 0.676577
[180]	training's auc: 0.846344	valid_1's auc: 0.677111
[190]	training's auc: 0.848538	valid_1's auc: 0.67753
[200]	training's auc: 0.850411	valid_1's auc: 0.677833
[210]	training's auc: 0.852178	valid_1's auc: 0.677986
[220]	training's auc: 0.853781	valid_1's auc: 0.678221
[230]	training's auc: 0.855588	valid_1's auc: 0.678499
[240]	training's auc: 0.856964	valid_1's auc: 0.678501
[250]	training's auc: 0.858251	valid_1's auc: 0.678576
[260]	training's auc: 0.859452	valid_1's auc: 0.678625
[270]	training's auc: 0.860761	valid_1's auc: 0.67867
[280]	training's auc: 0.861634	valid_1's auc: 0.678703
[290]	training's auc: 0.862696	valid_1's auc: 0.67868
[300]	training's auc: 0.863797	valid_1's auc: 0.678694
[310]	training's auc: 0.864782	valid_1's auc: 0.678664
Early stopping, best iteration is:
[295]	training's auc: 0.863256	valid_1's auc: 0.678708
done

Process finished with exit code 0
'''