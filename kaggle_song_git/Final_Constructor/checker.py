import sys
sys.path.insert(0, '../')
from me import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'

load_name = 'final_test_real.csv'
# load_name = 'final_train_play.csv'
# load_name = 'final_train_real.csv'
load_name = '[]_6360_cat__1513344488.csv'
df = read_df(load_name,
             read_from='../saves/submission/')
show_df(df, detail=True)

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


