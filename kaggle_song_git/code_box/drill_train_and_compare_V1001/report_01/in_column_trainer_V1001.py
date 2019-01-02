import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle
from sklearn.model_selection import train_test_split

since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt
print('What we got:')
print(df.dtypes)
print('number of columns:', len(df.columns))

num_boost_round = 500000
early_stopping_rounds = 50
verbose_eval = 10
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'learning_rate': 0.02,
    'verbose': -1,
    'num_leaves': 300,

    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 2,
    # 'bagging_seed': 1,
    # 'feature_fraction': 0.8,
    # 'feature_fraction_seed': 1,
    'max_bin': 255,
    'max_depth': -1,
}
fixed = ['msno',
         'song_id',
         'target',
         'source_system_tab',
         'source_screen_name',
         'source_type',
         'language',
         'artist_name',
         ]

for w in df.columns:
    if w in fixed:
        pass
    else:
        print('working on:', w)
        toto = [i for i in fixed]
        toto.append(w)
        df = df[toto]

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype('category')

        print()
        print()
        print('After selection:')
        print(df.dtypes)
        print('number of columns:', len(df.columns))
        print()
        print()
        length = len(df)
        train_size = 0.76
        train_set = df.head(int(length*train_size))
        val_set = df.drop(train_set.index)

        del df
        train_set = train_set.sample(frac=1)
        X_tr = train_set.drop(['target'], axis=1)
        Y_tr = train_set['target'].values

        X_val = val_set.drop(['target'], axis=1)
        Y_val = val_set['target'].values

        del train_set, val_set

        t = len(Y_tr)
        t1 = sum(Y_tr)
        t0 = t - t1
        print('train size:', t, 'number of 1:', t1, 'number of 0:', t0)
        print('train: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
        t = len(Y_val)
        t1 = sum(Y_val)
        t0 = t - t1
        print('val size:', t, 'number of 1:', t1, 'number of 0:', t0)
        print('val: 1 in all:', t1/t, '0 in all:', t0/t, '1/0:', t1/t0)
        print()
        print()

        train_set = lgb.Dataset(X_tr, Y_tr)
        val_set = lgb.Dataset(X_val, Y_val)
        del X_tr, Y_tr, X_val, Y_val

        print('Training...')

        model = lgb.train(params,
                          train_set,
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          valid_sets=val_set,
                          verbose_eval=verbose_eval,
                          )
        print('best score:', model.best_score['valid_0']['auc'])
        print('best iteration:', model.best_iteration)
        del train_set, val_set
        print('complete on:', w)
        print()
        dt = pickle.load(open(save_dir + load_name + '_dict.save', "rb"))
        df = pd.read_csv(save_dir + load_name + ".csv", dtype=dt)
        del dt


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 /media/ray/SSD/workspace/python/projects/kaggle_song_git/fake_train_and_compare_V1001/in_column_trainer_V1001.py
What we got:
msno                            object
song_id                         object
source_system_tab               object
source_screen_name              object
source_type                     object
target                           uint8
artist_name                     object
language                      category
fake_song_count                  int64
fake_liked_song_count            int64
fake_disliked_song_count         int64
fake_artist_count                int64
fake_liked_artist_count          int64
fake_disliked_artist_count       int64
fake_member_count                int64
fake_member_song_count           int64
fake_disliked_member_count       int64
dtype: object
number of columns: 17
working on: fake_song_count


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_song_count          int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:662: UserWarning: categorical_feature in param dict is overrided.
  warnings.warn('categorical_feature in param dict is overrided.')
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.635128
[20]	valid_0's auc: 0.63675
[30]	valid_0's auc: 0.638305
[40]	valid_0's auc: 0.63988
[50]	valid_0's auc: 0.641289
[60]	valid_0's auc: 0.64225
[70]	valid_0's auc: 0.643274
[80]	valid_0's auc: 0.644293
[90]	valid_0's auc: 0.645067
[100]	valid_0's auc: 0.645865
[110]	valid_0's auc: 0.646822
[120]	valid_0's auc: 0.64774
[130]	valid_0's auc: 0.648668
[140]	valid_0's auc: 0.649545
[150]	valid_0's auc: 0.650401
[160]	valid_0's auc: 0.651176
[170]	valid_0's auc: 0.652093
[180]	valid_0's auc: 0.652932
[190]	valid_0's auc: 0.65374
[200]	valid_0's auc: 0.654257
[210]	valid_0's auc: 0.654662
[220]	valid_0's auc: 0.655084
[230]	valid_0's auc: 0.655538
[240]	valid_0's auc: 0.655785
[250]	valid_0's auc: 0.656091
[260]	valid_0's auc: 0.656247
[270]	valid_0's auc: 0.656513
[280]	valid_0's auc: 0.656614
[290]	valid_0's auc: 0.656752
[300]	valid_0's auc: 0.656833
[310]	valid_0's auc: 0.656954
[320]	valid_0's auc: 0.657009
[330]	valid_0's auc: 0.657038
[340]	valid_0's auc: 0.657131
[350]	valid_0's auc: 0.657211
[360]	valid_0's auc: 0.657233
[370]	valid_0's auc: 0.657216
[380]	valid_0's auc: 0.65734
[390]	valid_0's auc: 0.657366
[400]	valid_0's auc: 0.657409
[410]	valid_0's auc: 0.657388
[420]	valid_0's auc: 0.657419
[430]	valid_0's auc: 0.657445
[440]	valid_0's auc: 0.657435
[450]	valid_0's auc: 0.657385
[460]	valid_0's auc: 0.657435
[470]	valid_0's auc: 0.657411
[480]	valid_0's auc: 0.657408
Early stopping, best iteration is:
[432]	valid_0's auc: 0.657452
best score: 0.657452439663
best iteration: 432
complete on: fake_song_count

working on: fake_liked_song_count


After selection:
msno                     category
song_id                  category
target                      uint8
source_system_tab        category
source_screen_name       category
source_type              category
language                 category
artist_name              category
fake_liked_song_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.612297
[20]	valid_0's auc: 0.614899
[30]	valid_0's auc: 0.616144
[40]	valid_0's auc: 0.617327
[50]	valid_0's auc: 0.618789
[60]	valid_0's auc: 0.620263
[70]	valid_0's auc: 0.621471
[80]	valid_0's auc: 0.622673
[90]	valid_0's auc: 0.623773
[100]	valid_0's auc: 0.624804
[110]	valid_0's auc: 0.625805
[120]	valid_0's auc: 0.626789
[130]	valid_0's auc: 0.627776
[140]	valid_0's auc: 0.628611
[150]	valid_0's auc: 0.629357
[160]	valid_0's auc: 0.630123
[170]	valid_0's auc: 0.630731
[180]	valid_0's auc: 0.631389
[190]	valid_0's auc: 0.631927
[200]	valid_0's auc: 0.632404
[210]	valid_0's auc: 0.632876
[220]	valid_0's auc: 0.633324
[230]	valid_0's auc: 0.633652
[240]	valid_0's auc: 0.633931
[250]	valid_0's auc: 0.634176
[260]	valid_0's auc: 0.634383
[270]	valid_0's auc: 0.634582
[280]	valid_0's auc: 0.634752
[290]	valid_0's auc: 0.634896
[300]	valid_0's auc: 0.634991
[310]	valid_0's auc: 0.635112
[320]	valid_0's auc: 0.635199
[330]	valid_0's auc: 0.635288
[340]	valid_0's auc: 0.635358
[350]	valid_0's auc: 0.635438
[360]	valid_0's auc: 0.635489
[370]	valid_0's auc: 0.635116
[380]	valid_0's auc: 0.635261
[390]	valid_0's auc: 0.635366
[400]	valid_0's auc: 0.635403
[410]	valid_0's auc: 0.635509
[420]	valid_0's auc: 0.635528
[430]	valid_0's auc: 0.635555
[440]	valid_0's auc: 0.63556
[450]	valid_0's auc: 0.635567
[460]	valid_0's auc: 0.635537
[470]	valid_0's auc: 0.635522
[480]	valid_0's auc: 0.63545
[490]	valid_0's auc: 0.635386
Early stopping, best iteration is:
[446]	valid_0's auc: 0.635601
best score: 0.635600759046
best iteration: 446
complete on: fake_liked_song_count

working on: fake_disliked_song_count


After selection:
msno                        category
song_id                     category
target                         uint8
source_system_tab           category
source_screen_name          category
source_type                 category
language                    category
artist_name                 category
fake_disliked_song_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.621913
[20]	valid_0's auc: 0.624109
[30]	valid_0's auc: 0.625942
[40]	valid_0's auc: 0.627349
[50]	valid_0's auc: 0.628561
[60]	valid_0's auc: 0.629582
[70]	valid_0's auc: 0.630419
[80]	valid_0's auc: 0.631473
[90]	valid_0's auc: 0.632298
[100]	valid_0's auc: 0.633142
[110]	valid_0's auc: 0.634154
[120]	valid_0's auc: 0.635058
[130]	valid_0's auc: 0.636004
[140]	valid_0's auc: 0.636986
[150]	valid_0's auc: 0.637905
[160]	valid_0's auc: 0.63875
[170]	valid_0's auc: 0.639782
[180]	valid_0's auc: 0.640672
[190]	valid_0's auc: 0.641351
[200]	valid_0's auc: 0.641899
[210]	valid_0's auc: 0.642374
[220]	valid_0's auc: 0.642803
[230]	valid_0's auc: 0.643184
[240]	valid_0's auc: 0.643521
[250]	valid_0's auc: 0.643821
[260]	valid_0's auc: 0.644099
[270]	valid_0's auc: 0.644357
[280]	valid_0's auc: 0.644546
[290]	valid_0's auc: 0.644716
[300]	valid_0's auc: 0.644891
[310]	valid_0's auc: 0.645005
[320]	valid_0's auc: 0.645174
[330]	valid_0's auc: 0.645472
[340]	valid_0's auc: 0.646094
[350]	valid_0's auc: 0.646522
[360]	valid_0's auc: 0.646577
[370]	valid_0's auc: 0.646631
[380]	valid_0's auc: 0.646734
[390]	valid_0's auc: 0.64688
[400]	valid_0's auc: 0.646969
[410]	valid_0's auc: 0.647013
[420]	valid_0's auc: 0.647174
[430]	valid_0's auc: 0.647259
[440]	valid_0's auc: 0.647308
[450]	valid_0's auc: 0.647353
[460]	valid_0's auc: 0.647382
[470]	valid_0's auc: 0.647432
[480]	valid_0's auc: 0.647448
[490]	valid_0's auc: 0.647469
[500]	valid_0's auc: 0.647469
[510]	valid_0's auc: 0.647562
[520]	valid_0's auc: 0.647576
[530]	valid_0's auc: 0.64759
[540]	valid_0's auc: 0.647641
[550]	valid_0's auc: 0.647652
[560]	valid_0's auc: 0.647682
[570]	valid_0's auc: 0.647689
[580]	valid_0's auc: 0.647696
[590]	valid_0's auc: 0.647684
[600]	valid_0's auc: 0.647695
[610]	valid_0's auc: 0.647686
[620]	valid_0's auc: 0.647695
[630]	valid_0's auc: 0.647713
[640]	valid_0's auc: 0.647712
[650]	valid_0's auc: 0.647697
[660]	valid_0's auc: 0.647692
[670]	valid_0's auc: 0.647678
[680]	valid_0's auc: 0.64767
[690]	valid_0's auc: 0.647652
Early stopping, best iteration is:
[643]	valid_0's auc: 0.647719
best score: 0.647719490257
best iteration: 643
complete on: fake_disliked_song_count

working on: fake_artist_count


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_artist_count        int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.64398
[20]	valid_0's auc: 0.646023
[30]	valid_0's auc: 0.648511
[40]	valid_0's auc: 0.649937
[50]	valid_0's auc: 0.651161
[60]	valid_0's auc: 0.652522
[70]	valid_0's auc: 0.653793
[80]	valid_0's auc: 0.655114
[90]	valid_0's auc: 0.656415
[100]	valid_0's auc: 0.657404
[110]	valid_0's auc: 0.658518
[120]	valid_0's auc: 0.659592
[130]	valid_0's auc: 0.66069
[140]	valid_0's auc: 0.661787
[150]	valid_0's auc: 0.662722
[160]	valid_0's auc: 0.663356
[170]	valid_0's auc: 0.664048
[180]	valid_0's auc: 0.664566
[190]	valid_0's auc: 0.665115
[200]	valid_0's auc: 0.665696
[210]	valid_0's auc: 0.666145
[220]	valid_0's auc: 0.66653
[230]	valid_0's auc: 0.666876
[240]	valid_0's auc: 0.667182
[250]	valid_0's auc: 0.667462
[260]	valid_0's auc: 0.667715
[270]	valid_0's auc: 0.668037
[280]	valid_0's auc: 0.668282
[290]	valid_0's auc: 0.668454
[300]	valid_0's auc: 0.668582
[310]	valid_0's auc: 0.668758
[320]	valid_0's auc: 0.668901
[330]	valid_0's auc: 0.669
[340]	valid_0's auc: 0.669146
[350]	valid_0's auc: 0.66921
[360]	valid_0's auc: 0.669281
[370]	valid_0's auc: 0.669458
[380]	valid_0's auc: 0.669503
[390]	valid_0's auc: 0.669522
[400]	valid_0's auc: 0.669558
[410]	valid_0's auc: 0.669666
[420]	valid_0's auc: 0.669693
[430]	valid_0's auc: 0.669728
[440]	valid_0's auc: 0.669758
[450]	valid_0's auc: 0.669787
[460]	valid_0's auc: 0.669827
[470]	valid_0's auc: 0.669882
[480]	valid_0's auc: 0.669903
[490]	valid_0's auc: 0.669921
[500]	valid_0's auc: 0.669926
[510]	valid_0's auc: 0.669935
[520]	valid_0's auc: 0.669925
[530]	valid_0's auc: 0.669964
[540]	valid_0's auc: 0.66997
[550]	valid_0's auc: 0.669961
[560]	valid_0's auc: 0.669971
[570]	valid_0's auc: 0.669968
[580]	valid_0's auc: 0.669966
Early stopping, best iteration is:
[534]	valid_0's auc: 0.669975
best score: 0.669975147237
best iteration: 534
complete on: fake_artist_count

working on: fake_liked_artist_count


After selection:
msno                       category
song_id                    category
target                        uint8
source_system_tab          category
source_screen_name         category
source_type                category
language                   category
artist_name                category
fake_liked_artist_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.639467
[20]	valid_0's auc: 0.642655
[30]	valid_0's auc: 0.644866
[40]	valid_0's auc: 0.64623
[50]	valid_0's auc: 0.647665
[60]	valid_0's auc: 0.648901
[70]	valid_0's auc: 0.650293
[80]	valid_0's auc: 0.651336
[90]	valid_0's auc: 0.652621
[100]	valid_0's auc: 0.653637
[110]	valid_0's auc: 0.654901
[120]	valid_0's auc: 0.655996
[130]	valid_0's auc: 0.656824
[140]	valid_0's auc: 0.657817
[150]	valid_0's auc: 0.658518
[160]	valid_0's auc: 0.65929
[170]	valid_0's auc: 0.660015
[180]	valid_0's auc: 0.66044
[190]	valid_0's auc: 0.660989
[200]	valid_0's auc: 0.661524
[210]	valid_0's auc: 0.662051
[220]	valid_0's auc: 0.662559
[230]	valid_0's auc: 0.662885
[240]	valid_0's auc: 0.663189
[250]	valid_0's auc: 0.663437
[260]	valid_0's auc: 0.663637
[270]	valid_0's auc: 0.663888
[280]	valid_0's auc: 0.664124
[290]	valid_0's auc: 0.664316
[300]	valid_0's auc: 0.664475
[310]	valid_0's auc: 0.664669
[320]	valid_0's auc: 0.664814
[330]	valid_0's auc: 0.664922
[340]	valid_0's auc: 0.665071
[350]	valid_0's auc: 0.665145
[360]	valid_0's auc: 0.665214
[370]	valid_0's auc: 0.665279
[380]	valid_0's auc: 0.665355
[390]	valid_0's auc: 0.665405
[400]	valid_0's auc: 0.665447
[410]	valid_0's auc: 0.665496
[420]	valid_0's auc: 0.665548
[430]	valid_0's auc: 0.665594
[440]	valid_0's auc: 0.665638
[450]	valid_0's auc: 0.665673
[460]	valid_0's auc: 0.665696
[470]	valid_0's auc: 0.665746
[480]	valid_0's auc: 0.665765
[490]	valid_0's auc: 0.66577
[500]	valid_0's auc: 0.665806
[510]	valid_0's auc: 0.665805
[520]	valid_0's auc: 0.665841
[530]	valid_0's auc: 0.665871
[540]	valid_0's auc: 0.665902
[550]	valid_0's auc: 0.665913
[560]	valid_0's auc: 0.66592
[570]	valid_0's auc: 0.665908
[580]	valid_0's auc: 0.665919
[590]	valid_0's auc: 0.665954
[600]	valid_0's auc: 0.665961
[610]	valid_0's auc: 0.665957
[620]	valid_0's auc: 0.665967
[630]	valid_0's auc: 0.665986
[640]	valid_0's auc: 0.665992
[650]	valid_0's auc: 0.665976
[660]	valid_0's auc: 0.665978
[670]	valid_0's auc: 0.665975
[680]	valid_0's auc: 0.666
[690]	valid_0's auc: 0.66601
[700]	valid_0's auc: 0.666011
[710]	valid_0's auc: 0.66603
[720]	valid_0's auc: 0.666038
[730]	valid_0's auc: 0.666064
[740]	valid_0's auc: 0.666071
[750]	valid_0's auc: 0.666065
[760]	valid_0's auc: 0.666057
[770]	valid_0's auc: 0.666057
[780]	valid_0's auc: 0.666057
Early stopping, best iteration is:
[735]	valid_0's auc: 0.666078
best score: 0.666078221153
best iteration: 735
complete on: fake_liked_artist_count

working on: fake_disliked_artist_count


After selection:
msno                          category
song_id                       category
target                           uint8
source_system_tab             category
source_screen_name            category
source_type                   category
language                      category
artist_name                   category
fake_disliked_artist_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.642233
[20]	valid_0's auc: 0.644817
[30]	valid_0's auc: 0.64695
[40]	valid_0's auc: 0.648592
[50]	valid_0's auc: 0.649777
[60]	valid_0's auc: 0.651023
[70]	valid_0's auc: 0.652344
[80]	valid_0's auc: 0.653768
[90]	valid_0's auc: 0.654942
[100]	valid_0's auc: 0.656011
[110]	valid_0's auc: 0.65709
[120]	valid_0's auc: 0.658017
[130]	valid_0's auc: 0.659099
[140]	valid_0's auc: 0.660055
[150]	valid_0's auc: 0.660774
[160]	valid_0's auc: 0.661395
[170]	valid_0's auc: 0.661867
[180]	valid_0's auc: 0.662459
[190]	valid_0's auc: 0.662941
[200]	valid_0's auc: 0.663346
[210]	valid_0's auc: 0.663615
[220]	valid_0's auc: 0.663853
[230]	valid_0's auc: 0.664155
[240]	valid_0's auc: 0.664419
[250]	valid_0's auc: 0.664681
[260]	valid_0's auc: 0.665043
[270]	valid_0's auc: 0.665258
[280]	valid_0's auc: 0.665516
[290]	valid_0's auc: 0.665631
[300]	valid_0's auc: 0.66582
[310]	valid_0's auc: 0.665948
[320]	valid_0's auc: 0.66606
[330]	valid_0's auc: 0.666186
[340]	valid_0's auc: 0.666238
[350]	valid_0's auc: 0.666312
[360]	valid_0's auc: 0.66642
[370]	valid_0's auc: 0.666546
[380]	valid_0's auc: 0.666594
[390]	valid_0's auc: 0.666626
[400]	valid_0's auc: 0.666689
[410]	valid_0's auc: 0.66674
[420]	valid_0's auc: 0.666798
[430]	valid_0's auc: 0.666866
[440]	valid_0's auc: 0.666891
[450]	valid_0's auc: 0.666906
[460]	valid_0's auc: 0.666935
[470]	valid_0's auc: 0.666972
[480]	valid_0's auc: 0.667017
[490]	valid_0's auc: 0.667026
[500]	valid_0's auc: 0.667065
[510]	valid_0's auc: 0.667107
[520]	valid_0's auc: 0.667119
[530]	valid_0's auc: 0.667124
[540]	valid_0's auc: 0.667135
[550]	valid_0's auc: 0.667126
[560]	valid_0's auc: 0.667149
[570]	valid_0's auc: 0.667163
[580]	valid_0's auc: 0.667148
[590]	valid_0's auc: 0.667153
[600]	valid_0's auc: 0.667166
[610]	valid_0's auc: 0.667169
[620]	valid_0's auc: 0.667162
[630]	valid_0's auc: 0.667176
[640]	valid_0's auc: 0.667164
[650]	valid_0's auc: 0.667172
[660]	valid_0's auc: 0.667193
[670]	valid_0's auc: 0.667153
[680]	valid_0's auc: 0.66714
[690]	valid_0's auc: 0.667129
[700]	valid_0's auc: 0.66713
[710]	valid_0's auc: 0.667143
Early stopping, best iteration is:
[661]	valid_0's auc: 0.667196
best score: 0.667195864603
best iteration: 661
complete on: fake_disliked_artist_count

working on: fake_member_count


After selection:
msno                  category
song_id               category
target                   uint8
source_system_tab     category
source_screen_name    category
source_type           category
language              category
artist_name           category
fake_member_count        int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.650904
[20]	valid_0's auc: 0.652145
[30]	valid_0's auc: 0.654287
[40]	valid_0's auc: 0.656197
[50]	valid_0's auc: 0.65776
[60]	valid_0's auc: 0.658756
[70]	valid_0's auc: 0.659912
[80]	valid_0's auc: 0.660871
[90]	valid_0's auc: 0.661914
[100]	valid_0's auc: 0.663008
[110]	valid_0's auc: 0.663891
[120]	valid_0's auc: 0.664775
[130]	valid_0's auc: 0.66599
[140]	valid_0's auc: 0.66698
[150]	valid_0's auc: 0.667845
[160]	valid_0's auc: 0.668491
[170]	valid_0's auc: 0.669016
[180]	valid_0's auc: 0.669493
[190]	valid_0's auc: 0.669933
[200]	valid_0's auc: 0.670247
[210]	valid_0's auc: 0.670497
[220]	valid_0's auc: 0.670969
[230]	valid_0's auc: 0.671356
[240]	valid_0's auc: 0.671706
[250]	valid_0's auc: 0.671893
[260]	valid_0's auc: 0.67207
[270]	valid_0's auc: 0.672205
[280]	valid_0's auc: 0.672323
[290]	valid_0's auc: 0.672566
[300]	valid_0's auc: 0.672777
[310]	valid_0's auc: 0.672951
[320]	valid_0's auc: 0.673033
[330]	valid_0's auc: 0.673161
[340]	valid_0's auc: 0.673257
[350]	valid_0's auc: 0.673398
[360]	valid_0's auc: 0.673466
[370]	valid_0's auc: 0.673602
[380]	valid_0's auc: 0.67369
[390]	valid_0's auc: 0.673784
[400]	valid_0's auc: 0.673877
[410]	valid_0's auc: 0.673993
[420]	valid_0's auc: 0.674058
[430]	valid_0's auc: 0.674124
[440]	valid_0's auc: 0.674193
[450]	valid_0's auc: 0.674245
[460]	valid_0's auc: 0.674291
[470]	valid_0's auc: 0.674325
[480]	valid_0's auc: 0.674365
[490]	valid_0's auc: 0.674404
[500]	valid_0's auc: 0.674441
[510]	valid_0's auc: 0.674452
[520]	valid_0's auc: 0.674471
[530]	valid_0's auc: 0.67451
[540]	valid_0's auc: 0.674547
[550]	valid_0's auc: 0.674574
[560]	valid_0's auc: 0.674584
[570]	valid_0's auc: 0.674597
[580]	valid_0's auc: 0.674587
[590]	valid_0's auc: 0.674593
[600]	valid_0's auc: 0.674585
[610]	valid_0's auc: 0.674592
[620]	valid_0's auc: 0.674605
[630]	valid_0's auc: 0.674611
[640]	valid_0's auc: 0.674631
[650]	valid_0's auc: 0.674639
[660]	valid_0's auc: 0.674621
[670]	valid_0's auc: 0.674634
[680]	valid_0's auc: 0.674626
[690]	valid_0's auc: 0.674618
Early stopping, best iteration is:
[646]	valid_0's auc: 0.674643
best score: 0.67464329168
best iteration: 646
complete on: fake_member_count

working on: fake_member_song_count


After selection:
msno                      category
song_id                   category
target                       uint8
source_system_tab         category
source_screen_name        category
source_type               category
language                  category
artist_name               category
fake_member_song_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.650052
[20]	valid_0's auc: 0.651636
[30]	valid_0's auc: 0.653641
[40]	valid_0's auc: 0.655396
[50]	valid_0's auc: 0.65708
[60]	valid_0's auc: 0.658291
[70]	valid_0's auc: 0.659384
[80]	valid_0's auc: 0.66039
[90]	valid_0's auc: 0.661387
[100]	valid_0's auc: 0.662469
[110]	valid_0's auc: 0.663614
[120]	valid_0's auc: 0.664245
[130]	valid_0's auc: 0.665211
[140]	valid_0's auc: 0.666084
[150]	valid_0's auc: 0.66686
[160]	valid_0's auc: 0.667479
[170]	valid_0's auc: 0.667982
[180]	valid_0's auc: 0.668473
[190]	valid_0's auc: 0.668824
[200]	valid_0's auc: 0.66928
[210]	valid_0's auc: 0.669699
[220]	valid_0's auc: 0.670073
[230]	valid_0's auc: 0.670501
[240]	valid_0's auc: 0.67085
[250]	valid_0's auc: 0.671069
[260]	valid_0's auc: 0.671305
[270]	valid_0's auc: 0.671562
[280]	valid_0's auc: 0.67181
[290]	valid_0's auc: 0.672026
[300]	valid_0's auc: 0.672166
[310]	valid_0's auc: 0.672347
[320]	valid_0's auc: 0.672513
[330]	valid_0's auc: 0.672624
[340]	valid_0's auc: 0.67275
[350]	valid_0's auc: 0.672845
[360]	valid_0's auc: 0.672947
[370]	valid_0's auc: 0.673007
[380]	valid_0's auc: 0.673075
[390]	valid_0's auc: 0.673247
[400]	valid_0's auc: 0.673301
[410]	valid_0's auc: 0.673383
[420]	valid_0's auc: 0.673445
[430]	valid_0's auc: 0.673472
[440]	valid_0's auc: 0.67355
[450]	valid_0's auc: 0.673606
[460]	valid_0's auc: 0.673661
[470]	valid_0's auc: 0.673696
[480]	valid_0's auc: 0.673743
[490]	valid_0's auc: 0.673758
[500]	valid_0's auc: 0.673794
[510]	valid_0's auc: 0.673835
[520]	valid_0's auc: 0.673873
[530]	valid_0's auc: 0.673875
[540]	valid_0's auc: 0.67391
[550]	valid_0's auc: 0.673904
[560]	valid_0's auc: 0.673897
[570]	valid_0's auc: 0.673891
[580]	valid_0's auc: 0.673914
[590]	valid_0's auc: 0.67392
[600]	valid_0's auc: 0.673939
[610]	valid_0's auc: 0.673959
[620]	valid_0's auc: 0.673952
[630]	valid_0's auc: 0.673933
[640]	valid_0's auc: 0.673942
[650]	valid_0's auc: 0.673951
Early stopping, best iteration is:
[606]	valid_0's auc: 0.67397
best score: 0.673969519205
best iteration: 606
complete on: fake_member_song_count

working on: fake_disliked_member_count


After selection:
msno                          category
song_id                       category
target                           uint8
source_system_tab             category
source_screen_name            category
source_type                   category
language                      category
artist_name                   category
fake_disliked_member_count       int64
dtype: object
number of columns: 9


train size: 5606837 number of 1: 2994894 number of 0: 2611943
train: 1 in all: 0.534150359641 0 in all: 0.465849640359 1/0: 1.14661537407
val size: 1770581 number of 1: 719762 number of 0: 1050819
val: 1 in all: 0.406511760829 0 in all: 0.593488239171 1/0: 0.68495335543


Training...
Training until validation scores don't improve for 50 rounds.
[10]	valid_0's auc: 0.649942
[20]	valid_0's auc: 0.652036
[30]	valid_0's auc: 0.653874
[40]	valid_0's auc: 0.655537
[50]	valid_0's auc: 0.657148
[60]	valid_0's auc: 0.658206
[70]	valid_0's auc: 0.659378
[80]	valid_0's auc: 0.660372
[90]	valid_0's auc: 0.661397
[100]	valid_0's auc: 0.662746
[110]	valid_0's auc: 0.663641
[120]	valid_0's auc: 0.664362
[130]	valid_0's auc: 0.66549
[140]	valid_0's auc: 0.666362
[150]	valid_0's auc: 0.667101
[160]	valid_0's auc: 0.667708
[170]	valid_0's auc: 0.668271
[180]	valid_0's auc: 0.668656
[190]	valid_0's auc: 0.669053
[200]	valid_0's auc: 0.669333
[210]	valid_0's auc: 0.669706
[220]	valid_0's auc: 0.67014
[230]	valid_0's auc: 0.670463
[240]	valid_0's auc: 0.670662
[250]	valid_0's auc: 0.670891
[260]	valid_0's auc: 0.671057
[270]	valid_0's auc: 0.671296
[280]	valid_0's auc: 0.671559
[290]	valid_0's auc: 0.671756
[300]	valid_0's auc: 0.671942
[310]	valid_0's auc: 0.672139
[320]	valid_0's auc: 0.672259
[330]	valid_0's auc: 0.6724
[340]	valid_0's auc: 0.672538
[350]	valid_0's auc: 0.672659
[360]	valid_0's auc: 0.672731
[370]	valid_0's auc: 0.67289
[380]	valid_0's auc: 0.672948
[390]	valid_0's auc: 0.673049
[400]	valid_0's auc: 0.67309
[410]	valid_0's auc: 0.673133
[420]	valid_0's auc: 0.673204
[430]	valid_0's auc: 0.673255
[440]	valid_0's auc: 0.673292
[450]	valid_0's auc: 0.67332
[460]	valid_0's auc: 0.67336
[470]	valid_0's auc: 0.673392
[480]	valid_0's auc: 0.673421
[490]	valid_0's auc: 0.673428
[500]	valid_0's auc: 0.673447
[510]	valid_0's auc: 0.673483
[520]	valid_0's auc: 0.673457
[530]	valid_0's auc: 0.67351
[540]	valid_0's auc: 0.673528
[550]	valid_0's auc: 0.673557
[560]	valid_0's auc: 0.673572
[570]	valid_0's auc: 0.673564
[580]	valid_0's auc: 0.673564
[590]	valid_0's auc: 0.673577
[600]	valid_0's auc: 0.673577
[610]	valid_0's auc: 0.673565
[620]	valid_0's auc: 0.673562
[630]	valid_0's auc: 0.673567
[640]	valid_0's auc: 0.673567
Early stopping, best iteration is:
[599]	valid_0's auc: 0.673577
best score: 0.673577166594
best iteration: 599
complete on: fake_disliked_member_count


[timer]: complete in 209m 52s

Process finished with exit code 0
'''