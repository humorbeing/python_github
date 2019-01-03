import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'
# save_dir = '../fake/'
if 'fake' in save_dir:
    print('-' * 45)
    print()
    print(' !' * 22)
    print()
    print('  this is fake world  ' * 2)
    print()
    print(' !' * 22)
    print()
    print('-' * 45)

load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt

print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of rows:', len(df))
print('number of columns:', len(df.columns))
print('<'*20)
# save_dir = '../fake/'
# # df = df[df['target'] == 1]
# # df = df[df['target'] == 0]
#
# count = {k: v for k, v in df['song_id'].value_counts().iteritems()}
# pickle.dump(count, open(save_dir+'total_song_count_dict.save', "wb"))
# # pickle.dump(count, open(save_dir+'liked_song_count_dict.save', "wb"))
# # pickle.dump(count, open(save_dir+'disliked_song_count_dict.save', "wb"))
# del count


kinds = {'0': 0}


def genre_id_count(x):
    global kinds
    if x == 'unknown':
        # kinds['0'] += 1
        return 0
    else:
        a = x.count('|') + 1
        if a == 1:
            if x in kinds:
                pass
            else:
                kinds[x] = 0
            kinds[x] += 1
        else:
            for i in x.split('|'):
                if i in kinds:
                    pass
                else:
                    kinds[i] = 0
                kinds[i] += 1
        return a


df['fake_genre_type_count'] = df['genre_ids'].apply(genre_id_count).astype(np.int64)

save_dir = '../fake/'
# df = df[df['target'] == 1]
# df = df[df['target'] == 0]
for i in kinds:
    print(i, ':', kinds[i])
# song_count = {k: v for k, v in df['song_id'].value_counts().iteritems()}
pickle.dump(kinds, open(save_dir+'total_single_genre_dict.save', "wb"))


def top1(x):
    if x == 'unknown':
        return '0'
    else:
        a = x.count('|') + 1
        if a == 1:
            return x
        else:
            top_1 = 0
            best_1 = 0
            for g in x.split('|'):
                if kinds[g] > best_1:
                    top_1 = g
                    best_1 = kinds[g]
            return top_1


df['fake_top1'] = df['genre_ids'].apply(top1).astype('category')


def top1_count(x):
    if x == '0':
        print(x)
        return 0
    else:
        return kinds[x]


df['fake_top1_count'] = df['fake_top1'].apply(top1_count).astype(np.int64)

se = set(df['fake_top1'])
print('len:', len(se))
print(se)
se = set(df['fake_top1_count'])
print('len:', len(se))
print(se)

# print(k)
if 'fake' in save_dir:
    print('-' * 45)
    print()
    print(' !' * 22)
    print()
    print('  this is fake world  ' * 2)
    print()
    print(' !' * 22)
    print()
    print('-' * 45)

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

