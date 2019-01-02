import pandas as pd
import time
import numpy as np
import pickle
import h2o
from sklearn.metrics import roc_auc_score

since = time.time()
h2o.init(nthreads=-1)

data_dir = '../data/'
save_dir = '../saves/'
read_from = save_dir

load_name = 'custom_members_fixed.csv'
load_name = 'custom_song_fixed.csv'
load_name = 'train_set.csv'
# load_name = 'test_set.csv'
load_name = 'train_best.csv'
load_name = load_name[:-4]

dt = pickle.load(open(read_from+load_name+'_dict.save', "rb"))
df = pd.read_csv(read_from+load_name+".csv", dtype=dt)
del dt

print()
print('>'*20)
print('>'*20)
print('dtypes of df:')

print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))
# print('<'*20)

# showme = True
showme = False
if showme:
    for on in df.columns:
        print()
        print('inspecting:'.ljust(20), on)
        # print('>'*20)
        print('any null:'.ljust(15), df[on].isnull().values.any())
        print('null number:'.ljust(15), df[on].isnull().values.sum())
        print(on, 'dtype:', df[on].dtypes)
        # print('describing', on, ':')
        # print(df[on].describe())
        print('-'*20)
        l = df[on]
        s = set(l)
        print('list len:'.ljust(20), len(l))
        print('set len:'.ljust(20), len(s))
        print()
print('<'*20)
print('<'*20)
print('<'*20)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')

print()
print('Our guest selection:')
print(df.dtypes)
print('number of columns:', len(df.columns))
print()

length = len(df)
train_size = 0.76
train_df = df.head(int(length*train_size))
val_df = df.drop(train_df.index)
del df


def auc(m, v, t):
    y_true = v[t]
    y_scores = m.predict(v)
    y_true = h2o.as_list(y_true, use_pandas=True).values
    y_scores = h2o.as_list(y_scores, use_pandas=True).values
    d = roc_auc_score(y_true, y_scores)
    return d


score = []

train_hf = h2o.H2OFrame(train_df)
# print(len(train_df))
# print(len(train_hf))
del train_df
val_hf = h2o.H2OFrame(val_df)
# print(len(val_df))
# print(len(val_hf))
# y_true = val_df['target'].values
del val_df

features = list(train_hf.columns)
features.remove('target')

# print(train_hf.head(5))
# print(val_hf.shape)


from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# Set up GLM for regression
glm_default = H2OGeneralizedLinearEstimator(
    family='gaussian',
    model_id='glm_default'
)

# Use .train() to build the model
glm_default.train(x=features,
                  y='target',
                  training_frame=train_hf)

score.append(auc(glm_default, val_hf, 'target'))
del glm_default
# auc(drf_default, val_hf, 'target')
# auc(gbm_default, val_hf, 'target')
# auc(dnn_default, val_hf, 'target')
# print(glm_default)
# print(glm_default.model_performance(val_hf))


from h2o.estimators.random_forest import H2ORandomForestEstimator

# Set up DRF for regression
# Add a seed for reproducibility
drf_default = H2ORandomForestEstimator(
    model_id='drf_default',
    seed=1234
)

# Use .train() to build the model
drf_default.train(x=features,
                  y='target',
                  training_frame=train_hf)

# auc(glm_default, val_hf, 'target')
score.append(auc(drf_default, val_hf, 'target'))
del drf_default
# auc(gbm_default, val_hf, 'target')
# auc(dnn_default, val_hf, 'target')

from h2o.estimators.gbm import H2OGradientBoostingEstimator

# Set up GBM for regression
# Add a seed for reproducibility
gbm_default = H2OGradientBoostingEstimator(
    model_id='gbm_default',
    seed=1234
)

# Use .train() to build the model
gbm_default.train(x=features,
                  y='target',
                  training_frame=train_hf)


# auc(glm_default, val_hf, 'target')
# auc(drf_default, val_hf, 'target')
score.append(auc(gbm_default, val_hf, 'target'))
del drf_default
# auc(dnn_default, val_hf, 'target')
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

# Set up DNN for regression
dnn_default = H2ODeepLearningEstimator(
    model_id='dnn_default'
)

# (not run) Change 'reproducible' to True if you want to reproduce the results
# The model will be built using a single thread (could be very slow)
# dnn_default = H2ODeepLearningEstimator(model_id = 'dnn_default', reproducible = True)

# Use .train() to build the model
dnn_default.train(x=features,
                  y='target',
                  training_frame=train_hf)


# auc(glm_default, val_hf, 'target')
# auc(drf_default, val_hf, 'target')
# auc(gbm_default, val_hf, 'target')
score.append(auc(dnn_default, val_hf, 'target'))
del dnn_default

for i in score:
    print('ACU:', i)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
