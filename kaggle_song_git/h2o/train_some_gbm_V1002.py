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
from me import *


since = time.time()
h2o.init(nthreads=-1)

data_dir = '../data/'
save_dir = '../saves/'
read_from = save_dir
load_name = 'train_best.csv'

df = read_df(read_from, load_name)
show_df(df)
train_df, val_df = div_df(df)
del df
score = []


train_hf = h2o.H2OFrame(train_df)
del train_df
val_hf = h2o.H2OFrame(val_df)
del val_df

features = list(train_hf.columns)
features.remove('target')

# gbm
gbm_default = H2OGradientBoostingEstimator(
    model_id='gbm_default',
    seed=1234
)

gbm_default.train(x=features,
                  y='target',
                  training_frame=train_hf)

auc(gbm_default, val_hf, 'target')

# manual
gbm_manual = H2OGradientBoostingEstimator(
    model_id='gbm_manual',
    seed=1234,
    ntrees=100,
    sample_rate=0.9,
    col_sample_rate=0.9
)


gbm_manual.train(x=features,
                 y='target',
                 training_frame=train_hf)

auc(gbm_manual, val_hf, 'target')

# cv
gbm_manual_cv = H2OGradientBoostingEstimator(
    model_id='gbm_manual_cv',
    seed=1234,
    ntrees=100,
    sample_rate=0.9,
    col_sample_rate=0.9,
    nfolds=5
)

gbm_manual_cv.train(x=features,
                    y='target',
                    training_frame=train_hf)

auc(gbm_manual_cv, val_hf, 'target')

# early stop
gbm_manual_cv_es = H2OGradientBoostingEstimator(
    model_id = 'gbm_manual_cv_es',
    seed = 1234,
    ntrees = 10000,   # increase the number of trees
    sample_rate = 0.9,
    col_sample_rate = 0.9,
    nfolds = 5,
    stopping_metric = 'mse', # let early stopping feature determine
    stopping_rounds = 15,     # the optimal number of trees
    score_tree_interval = 1
) # by looking at the MSE metric
# Use .train() to build the model

gbm_manual_cv_es.train(x = features,
                       y = 'target',
                       training_frame = train_hf)

auc(gbm_manual_cv_es, val_hf, 'target')

# grid search
from h2o.grid.grid_search import H2OGridSearch
search_criteria = {'strategy': "Cartesian"}
hyper_params = {'sample_rate': [0.7, 0.8, 0.9],
                'col_sample_rate': [0.7, 0.8, 0.9]}

gbm_full_grid = H2OGridSearch(
    H2OGradientBoostingEstimator(
        model_id = 'gbm_full_grid',
        seed = 1234,
        ntrees = 10000,
        nfolds = 5,
        stopping_metric = 'mse',
        stopping_rounds = 15,
        score_tree_interval = 1
    ),
    search_criteria = search_criteria, # full grid search
    hyper_params = hyper_params
)

gbm_full_grid.train(x = features,
                    y = 'target',
                    training_frame = train_hf)

gbm_full_grid_sorted = gbm_full_grid.get_grid(sort_by='mse', decreasing=False)
print(gbm_full_grid_sorted)

best_model_id = gbm_full_grid_sorted.model_ids[0]
best_gbm_from_full_grid = h2o.get_model(best_model_id)
print(best_gbm_from_full_grid.summary())

auc(best_gbm_from_full_grid, val_hf, 'target')

# all + random grid search
search_criteria = {'strategy': "RandomDiscrete",
                   'max_models': 9,
                   'seed': 1234}
hyper_params = {'sample_rate': [0.7, 0.8, 0.9],
                'col_sample_rate': [0.7, 0.8, 0.9],
                'max_depth': [3, 5, 7]}

gbm_rand_grid = H2OGridSearch(
                    H2OGradientBoostingEstimator(
                        model_id = 'gbm_rand_grid',
                        seed = 1234,
                        ntrees = 10000,
                        nfolds = 5,
                        stopping_metric = 'mse',
                        stopping_rounds = 15,
                        score_tree_interval = 1),
                    search_criteria = search_criteria, # full grid search
                    hyper_params = hyper_params)

gbm_rand_grid.train(x = features,
                    y = 'target',
                    training_frame = train_hf)

gbm_rand_grid_sorted = gbm_rand_grid.get_grid(sort_by='mse', decreasing=False)
print(gbm_rand_grid_sorted)

best_model_id = gbm_rand_grid_sorted.model_ids[0]
best_gbm_from_rand_grid = h2o.get_model(best_model_id)
print(best_gbm_from_rand_grid.summary())

auc(best_gbm_from_rand_grid, val_hf, 'target')

for i in score:
    print('ACU:', i)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
