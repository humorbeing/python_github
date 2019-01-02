import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


since = time.time()


data_dir = '../data/'
save_dir = '../saves/'


load_name = 'custom_song_fixed'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)
del dt


count = {}


def get_count1(x):
    try:
        return count[x]
    except KeyError:
        return 1


def get_count(x):
    try:
        return count[x]
    except KeyError:
        return 0


count = pickle.load(open(save_dir + 'total_song_count_dict.save', "rb"))
df['song_count'] = df['song_id'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(save_dir + 'liked_song_count_dict.save', "rb"))
# df['liked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + 'disliked_song_count_dict.save', "rb"))
# df['disliked_song_count'] = df['song_id'].apply(get_count).astype(np.int64)


count = pickle.load(open(save_dir + 'total_artist_count_dict.save', "rb"))
df['artist_count'] = df['artist_name'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(save_dir + 'liked_artist_count_dict.save', "rb"))
# df['liked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + 'disliked_artist_count_dict.save', "rb"))
# df['disliked_artist_count'] = df['artist_name'].apply(get_count).astype(np.int64)


# count = pickle.load(open(save_dir + 'total_member_count_dict.save', "rb"))
# df['member_count'] = df['msno'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(save_dir + 'liked_member_count_dict.save', "rb"))
# df['liked_member_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(save_dir + 'disliked_member_count_dict.save', "rb"))
# df['disliked_member_count'] = df['song_id'].apply(get_count).astype(np.int64)


count = pickle.load(open(save_dir + 'total_language_count_dict.save', "rb"))
df['language_count'] = df['language'].apply(get_count1).astype(np.int64)
# count = pickle.load(open(storage + 'liked_member_count_dict.save', "rb"))
# df['fake_liked_member_count'] = df['song_id'].apply(get_count).astype(np.int64)
# count = pickle.load(open(storage + 'disliked_member_count_dict.save', "rb"))
# df['fake_disliked_member_count'] = df['song_id'].apply(get_count).astype(np.int64)
del count


print('creating custom member.')
save_name = 'custom_song_'
vers = 'fixed'
d = df.dtypes.to_dict()
# print(d)
print('dtypes of df:')
print('>'*20)
print(df.dtypes)
print('number of columns:', len(df.columns))
print('number of data:', len(df))
print('<'*20)
df.to_csv(save_dir+save_name+vers+'.csv', index=False)
pickle.dump(d, open(save_dir+save_name+vers+'_dict.save', "wb"))

print('done.')

print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

