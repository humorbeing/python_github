import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import surprise
import time

since = time.time()

train = pd.read_csv('train.csv')
print("Training Set Loaded")

algo = surprise.SVD()
reader = surprise.Reader(rating_scale=(0,1))
data = surprise.Dataset.load_from_df(train[['msno', 'song_id', 'target']].dropna(), reader)
trainset = data.build_full_trainset()
algo.train(trainset)
print("Done Training")


test = pd.read_csv('test.csv')
submit = []
for index, row in test.iterrows():
    est = algo.predict(row['msno'], row['song_id']).est
    submit.append((row['id'], est))
submit = pd.DataFrame(submit, columns=['id', 'target'])
submit.to_csv('submission_from_fkk_cf(svd)_V1001.csv', index=False)
print("Created submission.csv")

time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


