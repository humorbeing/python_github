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

train_hf = h2o.H2OFrame(train_df)
del train_df
val_hf = h2o.H2OFrame(val_df)
del val_df

features = list(train_hf.columns)
t = 'target'
features.remove(t)
train_hf[t] = train_hf[t].asfactor()
val_hf[t] = val_hf[t].asfactor()

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# Set up GLM for binary classification
glm_default = H2OGeneralizedLinearEstimator(
    family = 'binomial',
    model_id = 'glm_default'
)

# Use .train() to build the model
glm_default.train(x = features,
                  y = t,
                  training_frame = train_hf)

print(glm_default.model_performance(val_hf))
print(glm_default.summary())
# auc(glm_default, val_hf, t)


print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
