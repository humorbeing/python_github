import pandas as pd
import lightgbm as lgb
import time
import pickle
import numpy as np

print()
print('This is [no drill] training.')
print()
since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

# barebone = True
barebone = False
if barebone:
    ccc = [i for i in df.columns]
    ccc.remove('target')
    df.drop(ccc, axis=1, inplace=True)


inner = [
    '[0.67982]_0.6788_Light_gbdt_1512750240.csv',
    '[0.62259]_0.6246_Light_gbdt_1512859793.csv'
]
inner = False


def insert_this(on):
    global df
    on = on[:-4]
    df1 = pd.read_csv('../saves/feature/'+on+'.csv')
    df1.drop('id', axis=1, inplace=True)
    on = on[-10:]
    # print(on)
    df1.rename(columns={'target': on}, inplace=True)
    # print(df1.head(10))
    df = df.join(df1)
    del df1


counter = {}


def get_count(x):
    try:
        return counter[x]
    except KeyError:
        return 0


def add_this_counter_column(on_in):
    global counter, df
    read_from = '../saves/'
    counter = pickle.load(open(read_from+'counter/'+'ITC_'+on_in+'_dict.save', "rb"))
    df['ITC_'+on_in] = df[on_in].apply(get_count).astype(np.int64)
    # counter = pickle.load(open(read_from + 'counter/' + 'CC11_' + on_in + '_dict.save', "rb"))
    # df['CC11_' + on_in] = df[on_in].apply(get_count).astype(np.int64)
    # df.drop(on_in, axis=1, inplace=True)


# for col in cols:
#     print("'{}',".format(col))
    # add_this_counter_column(col)

cols = ['song_id', 'msno']
for col in cols:
    # print("'{}',".format(col))
    add_this_counter_column(col)

del counter


def log10me(x):
    return np.log10(x)


def log10me1(x):
    return np.log10(x+1)


def xxx(x):
    d = x / (x + 1)
    return d


for col in cols:
    colc = 'ITC_'+col
    # df[colc + '_log10'] = df[colc].apply(log10me).astype(np.float64)
    df[colc + '_log10_1'] = df[colc].apply(log10me1).astype(np.float64)
    # df[colc + '_x_1'] = df[colc].apply(xxx).astype(np.float64)
    # col1 = 'CC11_'+col
    # df['OinC_'+col] = df[col1]/df[colc]
    # df.drop(colc, axis=1, inplace=True)


if inner:
    for i in inner:
        insert_this(i)


print('What we got:')
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))

num_boost_round = 995
estimate = 0.6887  # make sure put in something here

boosting = 'gbdt'

learning_rate = 0.02
num_leaves = 511
max_depth = -1

max_bin = 255
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

    'max_bin': max_bin,
    'lambda_l1': lambda_l1,
    'lambda_l2': lambda_l2,

    'bagging_fraction': bagging_fraction,
    'bagging_freq': bagging_freq,
    'bagging_seed': bagging_seed,
    'feature_fraction': feature_fraction,
    'feature_fraction_seed': feature_fraction_seed,
}
on = [
    'msno',
    'song_id',
    'target',
    'source_system_tab',
    'source_screen_name',
    'source_type',

    'artist_name',
    'ITC_song_id_log10_1',
    'ITC_msno_log10_1',
    'song_year',
    'top3_in_song',
]
df = df[on]

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')

print()
print('Our guest selection:')
print(df.dtypes)
print('number of columns:', len(df.columns))
print()

X_tr = df.drop(['target'], axis=1)
Y_tr = df['target'].values
del df

train_set = lgb.Dataset(X_tr, Y_tr)
train_set.max_bin = max_bin
del X_tr, Y_tr

params['metric'] = 'auc'
params['verbose'] = -1
params['objective'] = 'binary'

print('Training...')
print()
model = lgb.train(params,
                  train_set,
                  num_boost_round=num_boost_round,
                  )
del train_set

model_time = str(int(time.time()))
model_name = '_Light_'+boosting
model_name = '[]_'+str(estimate)+model_name
model_name = model_name + '_' + model_time
pickle.dump(model, open(save_dir+'model/'+model_name+'.model', "wb"))
print('model saved as: ', save_dir+'model/'+model_name+'.model')
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


