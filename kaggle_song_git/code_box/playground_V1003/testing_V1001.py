import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle


since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'test_fillna3'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)

del dt


# print("Train test and validation sets")

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')
        # test[col] = test[col].astype('category')

print(df.dtypes)
X_test = df.drop(['id'], axis=1)
ids = df['id'].values

del df
print('loading.')
model = pickle.load(open(save_dir+'model_V1001.save', "rb"))
print('done.')
print('Making predictions')
p_test_1 = model.predict(X_test)
del model
# p_test_2 = model_f2.predict(X_test)
# p_test_avg = np.mean([p_test_1, p_test_2], axis = 0)


print('Done making predictions')
print ('Saving predictions Model model of gbdt')

subm = pd.DataFrame()
subm['id'] = ids
del ids
subm['target'] = p_test_1
del p_test_1
subm.to_csv(save_dir + '611_submission_lgbm_avg.csv.gz',
            compression='gzip', index=False, float_format='%.5f')

print('Done!')
# pickle.dump(model_f1, open(save_dir+'model_V1001.save', "wb"))
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


