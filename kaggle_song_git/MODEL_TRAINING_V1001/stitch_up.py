import sys
sys.path.insert(0, '../')
from me import *
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import pickle

since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
read_from='../saves/submission/'
model_name = 'stitch_up_'
# ------------------------------------
load_name = '0.69411stitch_up_0.6904_[0.681_.csv'
model_name += load_name[:7]+'_'
w1 = 0.69411
df1 = pd.read_csv(read_from + load_name)

# ------------------------------------
load_name = '0.68993[]_6360_cat__1513461458.csv'
w2 = 0.68993
model_name += load_name[:7]+'_'
df2 = pd.read_csv(read_from + load_name)

# ------------------------------------
# load_name = '[0.68402]_0.6887_cat__1513383800.csv'
# w3 = 0.68402
# model_name += load_name[:6]+'_'
# df3 = pd.read_csv(read_from + load_name)

# p = df1['target']
p = np.zeros(shape=[len(df2)])
p += df1['target'] * w1
p += df2['target'] * w2
# p += df3['target'] * w3
p = p / (w1 + w2)

# p += df1['target']
# p += df2['target']
# p += df3['target']
# p = p / 3

print(df1.head())
print(df2.head())
# print(df3.head())
df = pd.DataFrame()
df['id'] = df2.id
df['target'] = p


print('-'*30)
print(df.head())

df.to_csv(save_dir+'submission/'+model_name+'.csv',
                index=False, float_format='%.5f')
print('[complete] submission name:', model_name+'.csv.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))