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
save_dir = '../fake/'
# df = df[df['target'] == 1]
# df = df[df['target'] == 0]

# song_count = {k: v for k, v in df['song_id'].value_counts().iteritems()}
# pickle.dump(song_count, open(save_dir+'total_song_count_dict.save', "wb"))
# # pickle.dump(song_count, open(save_dir+'liked_song_count_dict.save', "wb"))
# # pickle.dump(song_count, open(save_dir+'disliked_song_count_dict.save', "wb"))
# del song_count
# artist_count = {k: v for k, v in df['artist_name'].value_counts().iteritems()}
# pickle.dump(artist_count, open(save_dir+'total_artist_count_dict.save', "wb"))
# # pickle.dump(artist_count, open(save_dir+'liked_artist_count_dict.save', "wb"))
# # pickle.dump(artist_count, open(save_dir+'disliked_artist_count_dict.save', "wb"))
# del artist_count
# member_count = {k: v for k, v in df['msno'].value_counts().iteritems()}
# pickle.dump(member_count, open(save_dir+'total_member_count_dict.save', "wb"))
# # pickle.dump(member_count, open(save_dir+'liked_member_count_dict.save', "wb"))
# # pickle.dump(member_count, open(save_dir+'disliked_member_count_dict.save', "wb"))
# del member_count
#
# language_count = {k: v for k, v in df['language'].value_counts().iteritems()}
# pickle.dump(language_count, open(save_dir+'total_language_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'liked_language_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'disliked_language_count_dict.save', "wb"))
# del language_count
#
# language_count = {k: v for k, v in df['source_system_tab'].value_counts().iteritems()}
# pickle.dump(language_count, open(save_dir+'total_source_system_tab_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'liked_language_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'disliked_language_count_dict.save', "wb"))
# del language_count
#
# language_count = {k: v for k, v in df['source_screen_name'].value_counts().iteritems()}
# pickle.dump(language_count, open(save_dir+'total_source_screen_name_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'liked_language_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'disliked_language_count_dict.save', "wb"))
# del language_count
#
# language_count = {k: v for k, v in df['source_type'].value_counts().iteritems()}
# pickle.dump(language_count, open(save_dir+'total_source_type_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'liked_language_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'disliked_language_count_dict.save', "wb"))
# del language_count
#
# language_count = {k: v for k, v in df['genre_ids'].value_counts().iteritems()}
# pickle.dump(language_count, open(save_dir+'total_genre_ids_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'liked_language_count_dict.save', "wb"))
# # pickle.dump(language_count, open(save_dir+'disliked_language_count_dict.save', "wb"))
# del language_count

language_count = {k: v for k, v in df['song_year'].value_counts().iteritems()}
# language_count[0] = 0
pickle.dump(language_count, open(save_dir+'total_song_year_count_dict.save', "wb"))
# pickle.dump(language_count, open(save_dir+'liked_language_count_dict.save', "wb"))
# pickle.dump(language_count, open(save_dir+'disliked_language_count_dict.save', "wb"))
for i in language_count:
    print(i, ':', language_count[i])
del language_count

language_count = {k: v for k, v in df['song_country'].value_counts().iteritems()}
pickle.dump(language_count, open(save_dir+'total_song_country_count_dict.save', "wb"))
# pickle.dump(language_count, open(save_dir+'liked_language_count_dict.save', "wb"))
# pickle.dump(language_count, open(save_dir+'disliked_language_count_dict.save', "wb"))
language_count['0'] = 0
for i in language_count:
    print(i, ':', language_count[i])
del language_count

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

