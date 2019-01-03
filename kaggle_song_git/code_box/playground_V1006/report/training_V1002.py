import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
import time
import pickle
from sklearn.model_selection import train_test_split

since = time.time()

data_dir = '../data/'
save_dir = '../saves/'
load_name = 'train_set'
dt = pickle.load(open(save_dir+load_name+'_dict.save', "rb"))
df = pd.read_csv(save_dir+load_name+".csv", dtype=dt)

del dt
print(df.dtypes)
# df = df.drop(['song_count', 'liked_song_count',
#               'disliked_song_count', 'artist_count',
#               'liked_artist_count', 'disliked_artist_count'], axis=1)
df = df[['mn', 'sn', 'target']]
# df = df[['city', 'age', 'target']]
print("Train test and validation sets")

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category')
        # test[col] = test[col].astype('category')

print(df.dtypes)
# train = df.sample(frac=0.6, random_state=5)
# val = df.drop(train.index)
# print('df len: ', len(df))
# del df
X_train = df.drop(['target'], axis=1)
Y_train = df['target'].values
# X_val = val.drop(['target'], axis=1)
# Y_val = val['target'].values

# print('train len:', len(train))
# print('val len: ', len(val))
del df
# X_test = test.drop(['id'], axis=1)
# ids = test['id'].values
# X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)

# del train, test; gc.collect();

# train_set = lgb.Dataset(X_tr, y_tr)
# val_set = lgb.Dataset(X_val, y_val)


train_set = lgb.Dataset(X_train, Y_train,
                        categorical_feature=[0, 1],
                        )
print('Processed data...')
params = {'objective': 'binary',
          'metric': 'auc',
          # 'metric': 'binary_logloss',
          'boosting': 'gbdt',
          'learning_rate': 0.1,
          # 'verbosity': -1,
          'verbose': -1,
          # 'record': True,
          'num_leaves': 100,

          'bagging_fraction': 0.8,
          'bagging_freq': 2,
          'bagging_seed': 1,
          'feature_fraction': 0.8,
          'feature_fraction_seed': 1,
          'max_bin': 63,
          'max_depth': 10,
          # 'min_data': 500,
          'min_hessian': 0.05,
          # 'num_rounds': 500,
          # "min_data_in_leaf": 1,

          # 'lambda_l2': 0.5,
          # 'device': 'gpu',
          # 'gpu_platform_id': 0,
          # 'gpu_device_id': 0,
          # 'sparse_threshold': 1.0,
          # 'categorical_feature': (0,1,2,3),
         }
model = lgb.cv(params,
               train_set=train_set,
               num_boost_round=100,
               nfold=5,
               # feature_name=['mn', 'sn'],
               # categorical_feature='0,1',
               # early_stopping_rounds=10,
               #  valid_sets=val_set,

               verbose_eval=10,
               )
pickle.dump(model, open(save_dir+'model_V1001.save', "wb"))
print(model)
print(type(model))
print(len(model))
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


'''/usr/bin/python3.5 "/media/ray/SSD/workspace/python/projects/big data kaggle/playground_V1006/gbdt_random_V1001.py"
msno                        object
song_id                     object
source_system_tab           object
source_screen_name          object
source_type                 object
target                       uint8
city                         uint8
registered_via               uint8
mn                           int64
age                           int8
age_range                     int8
membership_days              int64
membership_days_range         int8
registration_year            int64
registration_month           int64
registration_date            int64
expiration_year              int64
expiration_month             int64
expiration_date              int64
sex                           int8
sex_guess                     int8
song_length                  int64
genre_ids                   object
artist_name                 object
composer                    object
lyricist                    object
language                      int8
sn                           int64
lyricists_count               int8
composer_count                int8
genre_ids_count               int8
length_range                 int64
length_bin_range             int64
length_chunk_range           int64
song_year                    int64
song_year_bin_range          int64
song_year_chunk_range        int64
song_country                object
rc                          object
artist_composer               int8
artist_composer_lyricist      int8
song_count                   int64
liked_song_count             int64
disliked_song_count          int64
artist_count                 int64
liked_artist_count           int64
disliked_artist_count        int64
dtype: object
Train test and validation sets
mn        int64
sn        int64
target    uint8
dtype: object
/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:1021: UserWarning: Using categorical_feature in Dataset.
  warnings.warn('Using categorical_feature in Dataset.')
Processed data...
[10]	cv_agg's auc: 0.586654 + 0.000297551
[20]	cv_agg's auc: 0.607948 + 0.000721776
[30]	cv_agg's auc: 0.621408 + 0.000943995
[40]	cv_agg's auc: 0.632319 + 0.000679517
[50]	cv_agg's auc: 0.640261 + 0.000587394
[60]	cv_agg's auc: 0.647493 + 0.000345615
[70]	cv_agg's auc: 0.653464 + 0.000384674
[80]	cv_agg's auc: 0.659172 + 0.000536495
[90]	cv_agg's auc: 0.663911 + 0.000440142
[100]	cv_agg's auc: 0.66778 + 0.000498698
{'auc-mean': [0.51597691094843978, 0.55973975165703471, 0.56471194826257987, 0.5682117575648451, 0.57278521480415745, 0.57549779767501763, 0.57933757677428177, 0.58135508862994079, 0.58469329905776424, 0.58665359975762832, 0.59009932283538646, 0.59168733752668357, 0.59435644663102472, 0.59615807633024986, 0.59857273476687367, 0.60023677978973911, 0.60266004329477296, 0.60420028302340678, 0.60661870440258991, 0.60794825208136993, 0.60992196618894279, 0.61105339593798147, 0.61303301587642867, 0.61380385238105062, 0.61577716644345626, 0.61655683312047382, 0.61828375762844578, 0.61889047222324955, 0.6207101131972218, 0.62140843667824919, 0.62319251671767262, 0.62368399414133047, 0.62527772453082509, 0.62587341343306568, 0.62738352239819495, 0.62795271953573339, 0.62950954928675196, 0.63017420167927851, 0.63160368907313469, 0.6323190777938732, 0.63357822417636955, 0.63402456888014336, 0.63538943982577378, 0.63576319023176642, 0.6370627411201083, 0.63729369196095642, 0.63841576661451449, 0.63877893951975184, 0.63992953295111943, 0.6402608160126565, 0.64137064582828485, 0.64175139002951542, 0.64296589960625417, 0.64332568501112797, 0.64436851355540792, 0.64453512149981695, 0.64563006758527142, 0.64615252380855215, 0.64725805316623597, 0.64749329956625423, 0.64859601541746292, 0.64885813441994888, 0.64977913914578334, 0.65010753883444705, 0.65104486282396523, 0.6511565609526706, 0.65215507164084019, 0.65232795732378768, 0.65324164677254415, 0.65346412394388576, 0.65448879511027358, 0.65464714462216311, 0.65557316594967707, 0.65589268096235709, 0.65667698747389613, 0.65699074042715555, 0.65775330849514035, 0.65800756549754302, 0.65881854568518872, 0.65917214119485501, 0.6599386928152644, 0.66016855538277253, 0.66099662445792529, 0.66122984842523291, 0.66188950375737254, 0.6621299973303062, 0.66290846105434209, 0.6630470658943034, 0.66370263625397163, 0.66391146869707451, 0.66457230099059095, 0.66467045722999929, 0.66539619121325155, 0.66549539348480535, 0.66619706700049242, 0.66629635824207545, 0.66695877123460967, 0.66703832956894604, 0.66764976099339246, 0.66777973841519089], 'auc-stdv': [0.00012160738646502979, 0.00057308978118913014, 0.0004534239997768885, 0.00044517290622631201, 0.00039771435210900246, 0.00041188304563292814, 0.0004133516696329106, 0.00017410198583485737, 0.00015979319288622658, 0.00029755082478805735, 0.00027361496373472439, 0.00011109773848993942, 0.00019939126585024865, 0.00050761866904675869, 0.00068541132823152229, 0.00063734978272294153, 0.00050524974274922078, 0.00047381362506977335, 0.00063011719073837423, 0.00072177642596311519, 0.00073121392669491967, 0.00080429583753900615, 0.00074308674798034217, 0.00078348125111368783, 0.00070488983050706947, 0.00082460250702000532, 0.00061593241086189043, 0.00065412051636257586, 0.00072925211830808599, 0.00094399478852067744, 0.00099683479691445938, 0.0010416309744501189, 0.0011211252458060957, 0.0012529677740850528, 0.001319105675134505, 0.0011627325663933642, 0.0010928591441040801, 0.00078333808292815822, 0.00088190653821111067, 0.00067951668092694939, 0.00056310470124597961, 0.00075583707880489846, 0.00089284488692949703, 0.00096896075070487147, 0.00087305520520164557, 0.00086838332224794309, 0.00079682111802862568, 0.00079105811024480761, 0.00079205682467101135, 0.00058739431670015501, 0.00067902089755947966, 0.0004744245549060375, 0.0004451723061348709, 0.00036212641616736798, 0.00040386200755995353, 0.0004508425745893819, 0.0003491043123696779, 0.00044484725085345921, 0.00041683192757123303, 0.00034561484171625963, 0.00037444710469647304, 0.00034879167244310146, 0.00044993561586635861, 0.00041035792716930371, 0.00041430860912787678, 0.00039348473924993236, 0.0003534104877531872, 0.00034433817371803667, 0.00031513679518444031, 0.00038467399407045993, 0.0002937855545579257, 0.00033356177429748676, 0.00042181616691475674, 0.00045833217775498905, 0.00048241415444819395, 0.0005437637929447376, 0.00050102164964013025, 0.00041721750622856108, 0.00048578244161620235, 0.00053649502947313032, 0.0005667345433062309, 0.00063813979241768239, 0.00066245274874713353, 0.00063491186122590618, 0.00067356628914787546, 0.00055125412778883436, 0.00042205995661739827, 0.00049514949102937283, 0.00043862620373163791, 0.00044014242017393007, 0.00039667039384704407, 0.00039486911291321979, 0.00040301898287456732, 0.0003749317398935124, 0.00046556552391520563, 0.00046605293455499269, 0.00044813731187176082, 0.00046096510396237726, 0.00045608358608483055, 0.00049869791622071495]}
<class 'dict'>
2

[timer]: complete in 5m 9s

Process finished with exit code 0
'''