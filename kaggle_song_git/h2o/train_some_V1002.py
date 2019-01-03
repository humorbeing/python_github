import pandas as pd
import time
import numpy as np
import pickle
import h2o
from sklearn.metrics import roc_auc_score
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

since = time.time()
h2o.init(nthreads=-1)

data_dir = '../data/'
save_dir = '../saves/'
read_from = save_dir

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

# showme = True
showme = False
if showme:
    for on in df.columns:
        print()
        print('inspecting:'.ljust(20), on)
        print('any null:'.ljust(15), df[on].isnull().values.any())
        print('null number:'.ljust(15), df[on].isnull().values.sum())
        print(on, 'dtype:', df[on].dtypes)
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
del train_df
val_hf = h2o.H2OFrame(val_df)
del val_df

features = list(train_hf.columns)
features.remove('target')

# glm
glm_default = H2OGeneralizedLinearEstimator(
    family='gaussian',
    model_id='glm_default'
)

glm_default.train(x=features,
                  y='target',
                  training_frame=train_hf)

score.append(auc(glm_default, val_hf, 'target'))
del glm_default

# drf
drf_default = H2ORandomForestEstimator(
    model_id='drf_default',
    seed=1234
)

drf_default.train(x=features,
                  y='target',
                  training_frame=train_hf)

score.append(auc(drf_default, val_hf, 'target'))
del drf_default

# gbm
gbm_default = H2OGradientBoostingEstimator(
    model_id='gbm_default',
    seed=1234
)

gbm_default.train(x=features,
                  y='target',
                  training_frame=train_hf)

score.append(auc(gbm_default, val_hf, 'target'))
del gbm_default

# dnn
dnn_default = H2ODeepLearningEstimator(
    model_id='dnn_default',
)

dnn_default.train(x=features,
                  y='target',
                  training_frame=train_hf)

score.append(auc(dnn_default, val_hf, 'target'))
del dnn_default

for i in score:
    print('ACU:', i)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
