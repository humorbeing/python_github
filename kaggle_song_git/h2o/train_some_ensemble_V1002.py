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
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch

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
t = 'target'
features.remove(t)
train_hf[t] = train_hf[t].asfactor()
val_hf[t] = val_hf[t].asfactor()

gbm1 = H2OGradientBoostingEstimator(
                        model_id = 'gbm1',
                        seed = 1234,
                        ntrees = 100,
                        nfolds = 2,
                        fold_assignment = "Modulo",               # needed for stacked ensembles
                        keep_cross_validation_predictions = True, # needed for stacked ensembles
                        # stopping_metric = 'mse',
                        # stopping_rounds = 15,
                        # score_tree_interval = 1
)

gbm2 = H2OGradientBoostingEstimator(
                        model_id = 'gbm2',
                        seed = 4321,
                        ntrees = 200,
                        nfolds = 2,
                        fold_assignment = "Modulo",               # needed for stacked ensembles
                        keep_cross_validation_predictions = True, # needed for stacked ensembles
                        # stopping_metric = 'mse',
                        # stopping_rounds = 15,
                        # score_tree_interval = 1
)

# glm1 = H2OGeneralizedLinearEstimator(
#     family = 'binomial',
#     model_id = 'glm1',
# nfolds = 5,
#                         fold_assignment = "Modulo",               # needed for stacked ensembles
#                         keep_cross_validation_predictions = True, # needed for stacked ensembles
#                         # stopping_metric = 'mse',
#                         # stopping_rounds = 15,
#                         # score_tree_interval = 1
# )

# glm2 = H2OGeneralizedLinearEstimator(
#     family = 'binomial',
#     model_id = 'glm2',
#     nfolds = 2,
#                         fold_assignment = "Modulo",               # needed for stacked ensembles
#                         keep_cross_validation_predictions = True, # needed for stacked ensembles
#                         # stopping_metric = 'mse',
#                         # stopping_rounds = 15,
#                         # score_tree_interval = 1
# )

gbm1.train(x = features,
                  y = t,
                  training_frame = train_hf)

gbm2.train(x = features,
                  y = t,
                  training_frame = train_hf)
#
# glm1.train(x = features,
#                   y = t,
#                   training_frame = train_hf)

# glm2.train(x = features,
#                   y = t,
#                   training_frame = train_hf)


all_ids = ['gbm1', 'gbm2']
ensemble = H2OStackedEnsembleEstimator(model_id = "my_ensemble",
                                       base_models = all_ids)

ensemble.train(x = features,
               y = t,
               training_frame = train_hf)


print(gbm1.model_performance(val_hf).auc())
print(gbm2.model_performance(val_hf).auc())
# print(glm1.model_performance(val_hf).auc())
# print(glm2.model_performance(val_hf).auc())
print(ensemble.model_performance(val_hf).auc())
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
