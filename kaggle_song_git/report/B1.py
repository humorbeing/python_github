
print()
'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/Final_LVL2/REAL_MORE_program.py
/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module '_catboost' does not match runtime version 3.5
  return f(*args, **kwds)

This is [no drill] training.

ON LEVEL: 2

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target               uint8
Ldrt_lvl2_1        float64
Lgos_lvl2_1        float64
Lrf_lvl2_1         float64
Lgbt_lvl2_1        float64
Ldrt_lvl_2         float64
Lgos_lvl2_2        float64
Lrf_lvl2_2         float64
Lgbt_lvl2_2        float64
nn_lvl2_1          float64
nn_lvl2_2          float64
rogi_lvl2_1        float64
sgd_lvl2           float64
GaussianNB_lvl2    float64
CV_lvl2            float64
RF_lvl2            float64
dtype: object
number of rows: 2459140
number of columns: 16

'target',
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                 category
Ldrt_lvl2_1         float64
Lgos_lvl2_1         float64
Lrf_lvl2_1          float64
Lgbt_lvl2_1         float64
Ldrt_lvl_2          float64
Lgos_lvl2_2         float64
Lrf_lvl2_2          float64
Lgbt_lvl2_2         float64
nn_lvl2_1           float64
nn_lvl2_2           float64
rogi_lvl2_1         float64
sgd_lvl2            float64
GaussianNB_lvl2     float64
CV_lvl2             float64
RF_lvl2             float64
dtype: object
number of rows: 2556790
number of columns: 16

'id',
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

in model: LogisticRegression  k-fold: 1 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression
0       1            0.487487
1       1            0.499615
2       1            0.841921
3       1            0.907815
4       0            0.843299
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # #
0.311923847443
0.355496601728
0.0754727749423
0.0592252500629
0.0511914665407
# # # # # # # # # #

in model: LogisticRegression  k-fold: 2 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression
0       1            0.843882
1       1            0.698659
2       1            0.748664
3       1            0.925250
4       1            0.879775
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # #
0.622446198416
0.710686088061
0.150980791158
0.118289810931
0.101991484372
# # # # # # # # # #

in model: LogisticRegression  k-fold: 3 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression
0       1            0.841676
1       1            0.839340
2       1            0.878845
3       1            0.915220
4       1            0.943338
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # #
0.93533518004
1.06814565051
0.224868866655
0.175999550074
0.15172135198
# # # # # # # # # #
  id  LogisticRegression
0  0            0.311778
1  1            0.356049
2  2            0.074956
3  3            0.058667
4  4            0.050574
AUC train 0.857661585884
AUC train 0.857520587864
AUC train 0.857539016708

in model: SGDClassifier  k-fold: 1 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
/usr/local/lib/python3.5/dist-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  "and default tol will be 1e-3." % type(self), FutureWarning)
   target  LogisticRegression  SGDClassifier
0       1            0.487487       0.491636
1       1            0.499615       0.501528
2       1            0.841921       0.847700
3       1            0.907815       0.897237
4       0            0.843299       0.832207
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # #
0.292220911597
0.329333990066
0.0796673391557
0.0664408717226
0.0588184934567
# # # # # # # # # #

in model: SGDClassifier  k-fold: 2 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier
0       1            0.843882       0.828006
1       1            0.698659       0.697486
2       1            0.748664       0.726070
3       1            0.925250       0.915613
4       1            0.879775       0.867366
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # #
0.584851586849
0.656962463767
0.159159646501
0.132309516075
0.116952091613
# # # # # # # # # #

in model: SGDClassifier  k-fold: 3 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier
0       1            0.841676       0.818027
1       1            0.839340       0.840298
2       1            0.878845       0.874208
3       1            0.915220       0.904389
4       1            0.943338       0.934844
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # #
0.881687775963
0.991052288622
0.239713814405
0.199432563923
0.176355675928
# # # # # # # # # #
  id  LogisticRegression  SGDClassifier
0  0            0.311778       0.293896
1  1            0.356049       0.330351
2  2            0.074956       0.079905
3  3            0.058667       0.066478
4  4            0.050574       0.058785
AUC train 0.857845634159
AUC train 0.857549441016
AUC train 0.85779136758

in model: GaussianNB  k-fold: 1 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.487487       0.491636    0.000523
1       1            0.499615       0.501528    0.982677
2       1            0.841921       0.847700    1.000000
3       1            0.907815       0.897237    1.000000
4       0            0.843299       0.832207    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # #
3.63952728868e-09
2.64958293076e-07
7.07926488685e-17
2.52513432177e-17
7.75857317941e-18
# # # # # # # # # #

in model: GaussianNB  k-fold: 2 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.843882       0.828006    1.000000
1       1            0.698659       0.697486    0.999999
2       1            0.748664       0.726070    1.000000
3       1            0.925250       0.915613    1.000000
4       1            0.879775       0.867366    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # #
7.62545646207e-09
5.52003687594e-07
1.53322897268e-16
5.48275455787e-17
1.69186204453e-17
# # # # # # # # # #

in model: GaussianNB  k-fold: 3 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.841676       0.818027         1.0
1       1            0.839340       0.840298         1.0
2       1            0.878845       0.874208         1.0
3       1            0.915220       0.904389         1.0
4       1            0.943338       0.934844         1.0
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # #
1.11179824983e-08
8.07573312357e-07
2.1779863522e-16
7.77304903322e-17
2.39144345484e-17
# # # # # # # # # #
  id  LogisticRegression  SGDClassifier    GaussianNB
0  0            0.311778       0.293896  3.705994e-09
1  1            0.356049       0.330351  2.691911e-07
2  2            0.074956       0.079905  7.259955e-17
3  3            0.058667       0.066478  2.591016e-17
4  4            0.050574       0.058785  7.971478e-18
AUC train 0.855546040834
AUC train 0.855217529623
AUC train 0.855600855714

in model: CV  k-fold: 1 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.487487       0.491636    0.000523  0.429143
1       1            0.499615       0.501528    0.982677  0.540024
2       1            0.841921       0.847700    1.000000  0.876499
3       1            0.907815       0.897237    1.000000  0.940648
4       0            0.843299       0.832207    1.000000  0.867283
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # #
0.30039847923
0.344869126523
0.0687647772685
0.0527302272226
0.0367802427788
# # # # # # # # # #

in model: CV  k-fold: 2 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.843882       0.828006    1.000000  0.873330
1       1            0.698659       0.697486    0.999999  0.634356
2       1            0.748664       0.726070    1.000000  0.687208
3       1            0.925250       0.915613    1.000000  0.941246
4       1            0.879775       0.867366    1.000000  0.941246
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # #
0.604173181164
0.696935817403
0.137152575335
0.10619519445
0.0730237635118
# # # # # # # # # #

in model: CV  k-fold: 3 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.841676       0.818027         1.0  0.857996
1       1            0.839340       0.840298         1.0  0.834006
2       1            0.878845       0.874208         1.0  0.940878
3       1            0.915220       0.904389         1.0  0.940878
4       1            0.943338       0.934844         1.0  0.940878
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # #
0.907547344136
1.04628703086
0.205983004017
0.160277074131
0.10863748209
# # # # # # # # # #
  id  LogisticRegression  SGDClassifier    GaussianNB        CV
0  0            0.311778       0.293896  3.705994e-09  0.302516
1  1            0.356049       0.330351  2.691911e-07  0.348762
2  2            0.074956       0.079905  7.259955e-17  0.068661
3  3            0.058667       0.066478  2.591016e-17  0.053426
4  4            0.050574       0.058785  7.971478e-18  0.036212
AUC train 0.855607437893
AUC train 0.85524570334
AUC train 0.855631112368

in model: RandomForest  k-fold: 1 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.487487       0.491636    0.000523  0.429143
1       1            0.499615       0.501528    0.982677  0.540024
2       1            0.841921       0.847700    1.000000  0.876499
3       1            0.907815       0.897237    1.000000  0.940648
4       0            0.843299       0.832207    1.000000  0.867283

   RandomForest
0      0.378274
1      0.590330
2      0.901567
3      0.901567
4      0.824077
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # #
0.378274201236
0.378274201236
0.124649105762
0.124649105762
0.124649105762
# # # # # # # # # #

in model: RandomForest  k-fold: 2 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.843882       0.828006    1.000000  0.873330
1       1            0.698659       0.697486    0.999999  0.634356
2       1            0.748664       0.726070    1.000000  0.687208
3       1            0.925250       0.915613    1.000000  0.941246
4       1            0.879775       0.867366    1.000000  0.941246

   RandomForest
0      0.899481
1      0.641046
2      0.641046
3      0.899481
4      0.899481
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # #
0.757858793743
0.757858793743
0.249427944181
0.249427944181
0.249427944181
# # # # # # # # # #

in model: RandomForest  k-fold: 3 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.841676       0.818027         1.0  0.857996
1       1            0.839340       0.840298         1.0  0.834006
2       1            0.878845       0.874208         1.0  0.940878
3       1            0.915220       0.904389         1.0  0.940878
4       1            0.943338       0.934844         1.0  0.940878

   RandomForest
0      0.901812
1      0.901812
2      0.901812
3      0.901812
4      0.901812
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # #
1.13821558967
1.13821558967
0.373944652758
0.373944652758
0.373944652758
# # # # # # # # # #
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest
0  0            0.311778       0.293896  3.705994e-09  0.302516      0.379405
1  1            0.356049       0.330351  2.691911e-07  0.348762      0.379405
2  2            0.074956       0.079905  7.259955e-17  0.068661      0.124648
3  3            0.058667       0.066478  2.591016e-17  0.053426      0.124648
4  4            0.050574       0.058785  7.971478e-18  0.036212      0.124648
AUC train 0.844566079119
AUC train 0.844600502306
AUC train 0.844433630538

in model: Neural_net  k-fold: 1 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.487487       0.491636    0.000523  0.429143
1       1            0.499615       0.501528    0.982677  0.540024
2       1            0.841921       0.847700    1.000000  0.876499
3       1            0.907815       0.897237    1.000000  0.940648
4       0            0.843299       0.832207    1.000000  0.867283

   RandomForest  Neural_net
0      0.378274    0.490138
1      0.590330    0.512203
2      0.901567    0.843207
3      0.901567    0.904837
4      0.824077    0.842152
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # #
0.314521966553
0.348630742917
0.0780767718639
0.0540933058719
0.037413335956
# # # # # # # # # #

in model: Neural_net  k-fold: 2 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.843882       0.828006    1.000000  0.873330
1       1            0.698659       0.697486    0.999999  0.634356
2       1            0.748664       0.726070    1.000000  0.687208
3       1            0.925250       0.915613    1.000000  0.941246
4       1            0.879775       0.867366    1.000000  0.941246

   RandomForest  Neural_net
0      0.899481    0.834469
1      0.641046    0.688104
2      0.641046    0.740652
3      0.899481    0.919028
4      0.899481    0.869331
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # #
0.622804856411
0.699319435253
0.154125969174
0.107065368654
0.0744599234206
# # # # # # # # # #

in model: Neural_net  k-fold: 3 / 3

- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.841676       0.818027         1.0  0.857996
1       1            0.839340       0.840298         1.0  0.834006
2       1            0.878845       0.874208         1.0  0.940878
3       1            0.915220       0.904389         1.0  0.940878
4       1            0.943338       0.934844         1.0  0.940878

   RandomForest  Neural_net
0      0.901812    0.840755
1      0.901812    0.834830
2      0.901812    0.874772
3      0.901812    0.911637
4      0.901812    0.939572
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # #
0.937222728981
1.04821582865
0.227801699446
0.159372097479
0.111923403908
# # # # # # # # # #
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.311778       0.293896  3.705994e-09  0.302516      0.379405
1  1            0.356049       0.330351  2.691911e-07  0.348762      0.379405
2  2            0.074956       0.079905  7.259955e-17  0.068661      0.124648
3  3            0.058667       0.066478  2.591016e-17  0.053426      0.124648
4  4            0.050574       0.058785  7.971478e-18  0.036212      0.124648

   Neural_net
0    0.312408
1    0.349405
2    0.075934
3    0.053124
4    0.037308
AUC train 0.857746576843
AUC train 0.857609942593
AUC train 0.857660461499

in model: DART  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.857512	valid_1's auc: 0.857578
[20]	training's auc: 0.857606	valid_1's auc: 0.857569
[30]	training's auc: 0.857638	valid_1's auc: 0.857534
[40]	training's auc: 0.857662	valid_1's auc: 0.857532
[50]	training's auc: 0.857695	valid_1's auc: 0.857551
[60]	training's auc: 0.857705	valid_1's auc: 0.857525
Early stopping, best iteration is:
[13]	training's auc: 0.857565	valid_1's auc: 0.857606
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.487487       0.491636    0.000523  0.429143
1       1            0.499615       0.501528    0.982677  0.540024
2       1            0.841921       0.847700    1.000000  0.876499
3       1            0.907815       0.897237    1.000000  0.940648
4       0            0.843299       0.832207    1.000000  0.867283

   RandomForest  Neural_net      DART
0      0.378274    0.490138  0.466508
1      0.590330    0.512203  0.524702
2      0.901567    0.843207  0.740205
3      0.901567    0.904837  0.827186
4      0.824077    0.842152  0.711476
# # # # # # # # # #
0.356125185679
0.392366295835
0.146618001094
0.142889381659
0.073801989667
# # # # # # # # # #

in model: DART  k-fold: 2 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.857626	valid_1's auc: 0.857295
[20]	training's auc: 0.857697	valid_1's auc: 0.857291
[30]	training's auc: 0.857738	valid_1's auc: 0.857218
[40]	training's auc: 0.857763	valid_1's auc: 0.857225
[50]	training's auc: 0.857801	valid_1's auc: 0.857252
[60]	training's auc: 0.857833	valid_1's auc: 0.857213
Early stopping, best iteration is:
[14]	training's auc: 0.85767	valid_1's auc: 0.85732
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.843882       0.828006    1.000000  0.873330
1       1            0.698659       0.697486    0.999999  0.634356
2       1            0.748664       0.726070    1.000000  0.687208
3       1            0.925250       0.915613    1.000000  0.941246
4       1            0.879775       0.867366    1.000000  0.941246

   RandomForest  Neural_net      DART
0      0.899481    0.834469  0.732231
1      0.641046    0.688104  0.626372
2      0.641046    0.740652  0.644807
3      0.899481    0.919028  0.846286
4      0.899481    0.869331  0.775205
# # # # # # # # # #
0.714510776599
0.79955789987
0.313667129929
0.288693536365
0.156769284929
# # # # # # # # # #

in model: DART  k-fold: 3 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.857549	valid_1's auc: 0.857515
[20]	training's auc: 0.857619	valid_1's auc: 0.857519
[30]	training's auc: 0.857653	valid_1's auc: 0.857507
[40]	training's auc: 0.857675	valid_1's auc: 0.857486
[50]	training's auc: 0.857716	valid_1's auc: 0.857532
[60]	training's auc: 0.85773	valid_1's auc: 0.857514
Early stopping, best iteration is:
[14]	training's auc: 0.857597	valid_1's auc: 0.857548
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.841676       0.818027         1.0  0.857996
1       1            0.839340       0.840298         1.0  0.834006
2       1            0.878845       0.874208         1.0  0.940878
3       1            0.915220       0.904389         1.0  0.940878
4       1            0.943338       0.934844         1.0  0.940878

   RandomForest  Neural_net      DART
0      0.901812    0.840755  0.696471
1      0.901812    0.834830  0.707039
2      0.901812    0.874772  0.772372
3      0.901812    0.911637  0.818925
4      0.901812    0.939572  0.888672
# # # # # # # # # #
1.07492509181
1.20587250958
0.481791765999
0.446940053213
0.23867368239
# # # # # # # # # #
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.311778       0.293896  3.705994e-09  0.302516      0.379405
1  1            0.356049       0.330351  2.691911e-07  0.348762      0.379405
2  2            0.074956       0.079905  7.259955e-17  0.068661      0.124648
3  3            0.058667       0.066478  2.591016e-17  0.053426      0.124648
4  4            0.050574       0.058785  7.971478e-18  0.036212      0.124648

   Neural_net      DART
0    0.312408  0.358308
1    0.349405  0.401958
2    0.075934  0.160597
3    0.053124  0.148980
4    0.037308  0.079558
AUC train 0.857591923146
AUC train 0.857308127546
AUC train 0.857541355904

in model: GOSS  k-fold: 1 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.85793	valid_1's auc: 0.858012
[20]	training's auc: 0.857912	valid_1's auc: 0.857923
[30]	training's auc: 0.857767	valid_1's auc: 0.857747
[40]	training's auc: 0.857679	valid_1's auc: 0.857579
[50]	training's auc: 0.857599	valid_1's auc: 0.857506
Early stopping, best iteration is:
[7]	training's auc: 0.857916	valid_1's auc: 0.858015
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.487487       0.491636    0.000523  0.429143
1       1            0.499615       0.501528    0.982677  0.540024
2       1            0.841921       0.847700    1.000000  0.876499
3       1            0.907815       0.897237    1.000000  0.940648
4       0            0.843299       0.832207    1.000000  0.867283

   RandomForest  Neural_net      DART      GOSS
0      0.378274    0.490138  0.466508  0.460126
1      0.590330    0.512203  0.524702  0.524705
2      0.901567    0.843207  0.740205  0.796747
3      0.901567    0.904837  0.827186  0.860924
4      0.824077    0.842152  0.711476  0.789397
# # # # # # # # # #
0.321463024418
0.358358898474
0.119856786382
0.108889459326
0.087947034739
# # # # # # # # # #

in model: GOSS  k-fold: 2 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.85806	valid_1's auc: 0.857774
[20]	training's auc: 0.858	valid_1's auc: 0.85762
[30]	training's auc: 0.85792	valid_1's auc: 0.857287
[40]	training's auc: 0.857759	valid_1's auc: 0.855746
[50]	training's auc: 0.85743	valid_1's auc: 0.85303
[60]	training's auc: 0.857503	valid_1's auc: 0.848371
Early stopping, best iteration is:
[12]	training's auc: 0.858064	valid_1's auc: 0.857775
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.843882       0.828006    1.000000  0.873330
1       1            0.698659       0.697486    0.999999  0.634356
2       1            0.748664       0.726070    1.000000  0.687208
3       1            0.925250       0.915613    1.000000  0.941246
4       1            0.879775       0.867366    1.000000  0.941246

   RandomForest  Neural_net      DART      GOSS
0      0.899481    0.834469  0.732231  0.813169
1      0.641046    0.688104  0.626372  0.666532
2      0.641046    0.740652  0.644807  0.706056
3      0.899481    0.919028  0.846286  0.924754
4      0.899481    0.869331  0.775205  0.848566
# # # # # # # # # #
0.62865901821
0.698275143775
0.204551043544
0.178043249507
0.133861509979
# # # # # # # # # #

in model: GOSS  k-fold: 3 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.857939	valid_1's auc: 0.857962
[20]	training's auc: 0.857931	valid_1's auc: 0.857894
[30]	training's auc: 0.857857	valid_1's auc: 0.85766
[40]	training's auc: 0.857609	valid_1's auc: 0.85758
[50]	training's auc: 0.857529	valid_1's auc: 0.856413
Early stopping, best iteration is:
[7]	training's auc: 0.857931	valid_1's auc: 0.85797
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.841676       0.818027         1.0  0.857996
1       1            0.839340       0.840298         1.0  0.834006
2       1            0.878845       0.874208         1.0  0.940878
3       1            0.915220       0.904389         1.0  0.940878
4       1            0.943338       0.934844         1.0  0.940878

   RandomForest  Neural_net      DART      GOSS
0      0.901812    0.840755  0.696471  0.786098
1      0.901812    0.834830  0.707039  0.782524
2      0.901812    0.874772  0.772372  0.824109
3      0.901812    0.911637  0.818925  0.864491
4      0.901812    0.939572  0.888672  0.905292
# # # # # # # # # #
0.95122817544
1.05840386043
0.328285394499
0.279629878328
0.22305048235
# # # # # # # # # #
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.311778       0.293896  3.705994e-09  0.302516      0.379405
1  1            0.356049       0.330351  2.691911e-07  0.348762      0.379405
2  2            0.074956       0.079905  7.259955e-17  0.068661      0.124648
3  3            0.058667       0.066478  2.591016e-17  0.053426      0.124648
4  4            0.050574       0.058785  7.971478e-18  0.036212      0.124648

   Neural_net      DART      GOSS
0    0.312408  0.358308  0.317076
1    0.349405  0.401958  0.352801
2    0.075934  0.160597  0.109428
3    0.053124  0.148980  0.093210
4    0.037308  0.079558  0.074350
AUC train 0.85801451636
AUC train 0.857774861803
AUC train 0.857969856844

in model: LIGHT_RF  k-fold: 1 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.858413	valid_1's auc: 0.857985
[20]	training's auc: 0.858441	valid_1's auc: 0.858016
[30]	training's auc: 0.85845	valid_1's auc: 0.858024
[40]	training's auc: 0.858455	valid_1's auc: 0.858026
[50]	training's auc: 0.858459	valid_1's auc: 0.858027
[60]	training's auc: 0.858463	valid_1's auc: 0.858028
[70]	training's auc: 0.858465	valid_1's auc: 0.85803
[80]	training's auc: 0.858466	valid_1's auc: 0.858031
[90]	training's auc: 0.858467	valid_1's auc: 0.858033
[100]	training's auc: 0.858466	valid_1's auc: 0.858032
[110]	training's auc: 0.858468	valid_1's auc: 0.858032
[120]	training's auc: 0.858468	valid_1's auc: 0.858032
[130]	training's auc: 0.858468	valid_1's auc: 0.858033
[140]	training's auc: 0.858468	valid_1's auc: 0.858033
[150]	training's auc: 0.858469	valid_1's auc: 0.858034
[160]	training's auc: 0.85847	valid_1's auc: 0.858033
[170]	training's auc: 0.85847	valid_1's auc: 0.858033
[180]	training's auc: 0.85847	valid_1's auc: 0.858033
[190]	training's auc: 0.858471	valid_1's auc: 0.858033
Early stopping, best iteration is:
[144]	training's auc: 0.858469	valid_1's auc: 0.858034
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.487487       0.491636    0.000523  0.429143
1       1            0.499615       0.501528    0.982677  0.540024
2       1            0.841921       0.847700    1.000000  0.876499
3       1            0.907815       0.897237    1.000000  0.940648
4       0            0.843299       0.832207    1.000000  0.867283

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF
0      0.378274    0.490138  0.466508  0.460126  0.450056
1      0.590330    0.512203  0.524702  0.524705  0.524408
2      0.901567    0.843207  0.740205  0.796747  0.785486
3      0.901567    0.904837  0.827186  0.860924  0.831164
4      0.824077    0.842152  0.711476  0.789397  0.776933
# # # # # # # # # #
0.319755756284
0.347363646965
0.157168727094
0.147669464176
0.140999618356
# # # # # # # # # #

in model: LIGHT_RF  k-fold: 2 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.858544	valid_1's auc: 0.857726
[20]	training's auc: 0.858574	valid_1's auc: 0.857755
[30]	training's auc: 0.858582	valid_1's auc: 0.857767
[40]	training's auc: 0.858589	valid_1's auc: 0.85777
[50]	training's auc: 0.858592	valid_1's auc: 0.857776
[60]	training's auc: 0.858596	valid_1's auc: 0.857777
[70]	training's auc: 0.858598	valid_1's auc: 0.857781
[80]	training's auc: 0.858599	valid_1's auc: 0.857782
[90]	training's auc: 0.8586	valid_1's auc: 0.857781
[100]	training's auc: 0.8586	valid_1's auc: 0.857782
[110]	training's auc: 0.858602	valid_1's auc: 0.857783
[120]	training's auc: 0.858602	valid_1's auc: 0.857783
[130]	training's auc: 0.858603	valid_1's auc: 0.857784
[140]	training's auc: 0.858603	valid_1's auc: 0.857782
[150]	training's auc: 0.858603	valid_1's auc: 0.857783
[160]	training's auc: 0.858603	valid_1's auc: 0.857784
[170]	training's auc: 0.858603	valid_1's auc: 0.857784
[180]	training's auc: 0.858604	valid_1's auc: 0.857784
[190]	training's auc: 0.858604	valid_1's auc: 0.857784
[200]	training's auc: 0.858604	valid_1's auc: 0.857784
[210]	training's auc: 0.858604	valid_1's auc: 0.857784
[220]	training's auc: 0.858604	valid_1's auc: 0.857785
[230]	training's auc: 0.858604	valid_1's auc: 0.857786
[240]	training's auc: 0.858604	valid_1's auc: 0.857786
[250]	training's auc: 0.858604	valid_1's auc: 0.857785
[260]	training's auc: 0.858604	valid_1's auc: 0.857785
[270]	training's auc: 0.858604	valid_1's auc: 0.857785
Early stopping, best iteration is:
[226]	training's auc: 0.858604	valid_1's auc: 0.857786
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.843882       0.828006    1.000000  0.873330
1       1            0.698659       0.697486    0.999999  0.634356
2       1            0.748664       0.726070    1.000000  0.687208
3       1            0.925250       0.915613    1.000000  0.941246
4       1            0.879775       0.867366    1.000000  0.941246

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF
0      0.899481    0.834469  0.732231  0.813169  0.779729
1      0.641046    0.688104  0.626372  0.666532  0.663895
2      0.641046    0.740652  0.644807  0.706056  0.698325
3      0.899481    0.919028  0.846286  0.924754  0.847301
4      0.899481    0.869331  0.775205  0.848566  0.812603
# # # # # # # # # #
0.63852576312
0.699632669292
0.313712579309
0.293484106652
0.278525471908
# # # # # # # # # #

in model: LIGHT_RF  k-fold: 3 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.858443	valid_1's auc: 0.857932
[20]	training's auc: 0.858474	valid_1's auc: 0.857964
[30]	training's auc: 0.858483	valid_1's auc: 0.857972
[40]	training's auc: 0.858488	valid_1's auc: 0.857976
[50]	training's auc: 0.85849	valid_1's auc: 0.857979
[60]	training's auc: 0.858492	valid_1's auc: 0.857984
[70]	training's auc: 0.858495	valid_1's auc: 0.857984
[80]	training's auc: 0.858496	valid_1's auc: 0.857982
[90]	training's auc: 0.858496	valid_1's auc: 0.857983
[100]	training's auc: 0.858497	valid_1's auc: 0.857983
[110]	training's auc: 0.858497	valid_1's auc: 0.857983
Early stopping, best iteration is:
[63]	training's auc: 0.858494	valid_1's auc: 0.857985
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.841676       0.818027         1.0  0.857996
1       1            0.839340       0.840298         1.0  0.834006
2       1            0.878845       0.874208         1.0  0.940878
3       1            0.915220       0.904389         1.0  0.940878
4       1            0.943338       0.934844         1.0  0.940878

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF
0      0.901812    0.840755  0.696471  0.786098  0.776875
1      0.901812    0.834830  0.707039  0.782524  0.777986
2      0.901812    0.874772  0.772372  0.824109  0.804917
3      0.901812    0.911637  0.818925  0.864491  0.834524
4      0.901812    0.939572  0.888672  0.905292  0.858481
# # # # # # # # # #
0.953086870418
1.0394709808
0.471967766112
0.44041499226
0.418807954787
# # # # # # # # # #
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.311778       0.293896  3.705994e-09  0.302516      0.379405
1  1            0.356049       0.330351  2.691911e-07  0.348762      0.379405
2  2            0.074956       0.079905  7.259955e-17  0.068661      0.124648
3  3            0.058667       0.066478  2.591016e-17  0.053426      0.124648
4  4            0.050574       0.058785  7.971478e-18  0.036212      0.124648

   Neural_net      DART      GOSS  LIGHT_RF
0    0.312408  0.358308  0.317076  0.317696
1    0.349405  0.401958  0.352801  0.346490
2    0.075934  0.160597  0.109428  0.157323
3    0.053124  0.148980  0.093210  0.146805
4    0.037308  0.079558  0.074350  0.139603
AUC train 0.858033731668
AUC train 0.857785766318
AUC train 0.857984557303

in model: LIGHTgbm  k-fold: 1 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.858172	valid_1's auc: 0.857996
[20]	training's auc: 0.858381	valid_1's auc: 0.857949
[30]	training's auc: 0.858579	valid_1's auc: 0.857882
[40]	training's auc: 0.858734	valid_1's auc: 0.85784
[50]	training's auc: 0.858926	valid_1's auc: 0.85779
Early stopping, best iteration is:
[8]	training's auc: 0.858125	valid_1's auc: 0.858
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.487487       0.491636    0.000523  0.429143
1       1            0.499615       0.501528    0.982677  0.540024
2       1            0.841921       0.847700    1.000000  0.876499
3       1            0.907815       0.897237    1.000000  0.940648
4       0            0.843299       0.832207    1.000000  0.867283

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm
0      0.378274    0.490138  0.466508  0.460126  0.450056  0.451865
1      0.590330    0.512203  0.524702  0.524705  0.524408  0.533759
2      0.901567    0.843207  0.740205  0.796747  0.785486  0.811919
3      0.901567    0.904837  0.827186  0.860924  0.831164  0.875311
4      0.824077    0.842152  0.711476  0.789397  0.776933  0.793054
# # # # # # # # # #
0.318707881921
0.346322760874
0.101824057266
0.0908997012774
0.0698013355065
# # # # # # # # # #

in model: LIGHTgbm  k-fold: 2 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.858292	valid_1's auc: 0.857746
[20]	training's auc: 0.8585	valid_1's auc: 0.857684
[30]	training's auc: 0.858671	valid_1's auc: 0.85753
[40]	training's auc: 0.85887	valid_1's auc: 0.857448
[50]	training's auc: 0.859047	valid_1's auc: 0.857409
Early stopping, best iteration is:
[6]	training's auc: 0.858192	valid_1's auc: 0.857755
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.843882       0.828006    1.000000  0.873330
1       1            0.698659       0.697486    0.999999  0.634356
2       1            0.748664       0.726070    1.000000  0.687208
3       1            0.925250       0.915613    1.000000  0.941246
4       1            0.879775       0.867366    1.000000  0.941246

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm
0      0.899481    0.834469  0.732231  0.813169  0.779729  0.790380
1      0.641046    0.688104  0.626372  0.666532  0.663895  0.648237
2      0.641046    0.740652  0.644807  0.706056  0.698325  0.688699
3      0.899481    0.919028  0.846286  0.924754  0.847301  0.881671
4      0.899481    0.869331  0.775205  0.848566  0.812603  0.817919
# # # # # # # # # #
0.650290773823
0.708434297127
0.228875748811
0.206901000528
0.167315792436
# # # # # # # # # #

in model: LIGHTgbm  k-fold: 3 / 3

Training until validation scores don't improve for 50 rounds.
[10]	training's auc: 0.858193	valid_1's auc: 0.857958
[20]	training's auc: 0.858398	valid_1's auc: 0.857908
[30]	training's auc: 0.858559	valid_1's auc: 0.857831
[40]	training's auc: 0.858748	valid_1's auc: 0.857773
[50]	training's auc: 0.858927	valid_1's auc: 0.857722
Early stopping, best iteration is:
[8]	training's auc: 0.858145	valid_1's auc: 0.857966
- - - - - - - - - -
'Ldrt_lvl2_1',
'Lgos_lvl2_1',
'Lrf_lvl2_1',
'Lgbt_lvl2_1',
'Ldrt_lvl_2',
'Lgos_lvl2_2',
'Lrf_lvl2_2',
'Lgbt_lvl2_2',
'nn_lvl2_1',
'nn_lvl2_2',
'rogi_lvl2_1',
'sgd_lvl2',
'GaussianNB_lvl2',
'CV_lvl2',
'RF_lvl2',
- - - - - - - - - -
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.841676       0.818027         1.0  0.857996
1       1            0.839340       0.840298         1.0  0.834006
2       1            0.878845       0.874208         1.0  0.940878
3       1            0.915220       0.904389         1.0  0.940878
4       1            0.943338       0.934844         1.0  0.940878

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm
0      0.901812    0.840755  0.696471  0.786098  0.776875  0.787781
1      0.901812    0.834830  0.707039  0.782524  0.777986  0.794892
2      0.901812    0.874772  0.772372  0.824109  0.804917  0.840223
3      0.901812    0.911637  0.818925  0.864491  0.834524  0.883226
4      0.901812    0.939572  0.888672  0.905292  0.858481  0.919545
# # # # # # # # # #
0.971778327832
1.0520942004
0.329588902668
0.295270798808
0.235858977784
# # # # # # # # # #
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.311778       0.293896  3.705994e-09  0.302516      0.379405
1  1            0.356049       0.330351  2.691911e-07  0.348762      0.379405
2  2            0.074956       0.079905  7.259955e-17  0.068661      0.124648
3  3            0.058667       0.066478  2.591016e-17  0.053426      0.124648
4  4            0.050574       0.058785  7.971478e-18  0.036212      0.124648

   Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm
0    0.312408  0.358308  0.317076  0.317696  0.323926
1    0.349405  0.401958  0.352801  0.346490  0.350698
2    0.075934  0.160597  0.109428  0.157323  0.109863
3    0.053124  0.148980  0.093210  0.146805  0.098424
4    0.037308  0.079558  0.074350  0.139603  0.078620
AUC train 0.857999590231
AUC train 0.857754689494
AUC train 0.857966470399

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of rows: 2556790
number of columns: 11

'id',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
 SAVE  SAVE  SAVE  SAVE  SAVE
 SAVE  SAVE  SAVE  SAVE  SAVE
 SAVE  SAVE  SAVE  SAVE  SAVE
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459140
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE
 SAVE  SAVE  SAVE  SAVE  SAVE
 SAVE  SAVE  SAVE  SAVE  SAVE
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE
 SAVE  SAVE  SAVE  SAVE  SAVE
 SAVE  SAVE  SAVE  SAVE  SAVE
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE
 SAVE  SAVE  SAVE  SAVE  SAVE
 SAVE  SAVE  SAVE  SAVE  SAVE
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of columns: 11
number of data: 2556790
<<<<<<<<<<<<<<<<<<<<
saving DONE.
ON LEVEL: 3

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of rows: 2459140
number of columns: 11

'target',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of rows: 2556790
number of columns: 11

'id',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

in model: LogisticRegression  k-fold: 1 / 3

- - - - - - - - - -
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - -

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
'''








'''/usr/bin/python3.5 /home/vb/workspace/python/kagglebigdata/Final_LVL2/REAL_MORE_program.py

/usr/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.4 of module '_catboost' does not match runtime version 3.5
This is [no drill] training.
  return f(*args, **kwds)

ON LEVEL: 3

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of rows: 2459140
number of columns: 11

'target',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of rows: 2556790
number of columns: 11

'id',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

in model: LogisticRegression  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.463085
1       1            0.565020
2       1            0.832866
3       1            0.909043
4       0            0.807145
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.295077344155
0.351776552723
0.0763387874797
0.0677219483808
0.0408428285118
# # # # # # # # # # 

in model: LogisticRegression  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.832084
1       1            0.685704
2       1            0.730518
3       1            0.935647
4       1            0.859949
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.589933024242
0.707128632465
0.151043599233
0.134237823251
0.0813341715641
# # # # # # # # # # 

in model: LogisticRegression  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.781120
1       1            0.787260
2       1            0.856362
3       1            0.903324
4       1            0.943478
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.884790061793
1.06044428549
0.228305042531
0.202938707624
0.123165820121
# # # # # # # # # # 
  id  LogisticRegression
0  0            0.294930
1  1            0.353481
2  2            0.076102
3  3            0.067646
4  4            0.041055
AUC train 0.857685780103
AUC train 0.857318538395
AUC train 0.857582237866

in model: SGDClassifier  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  "and default tol will be 1e-3." % type(self), FutureWarning)
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.463085       0.482879
1       1            0.565020       0.519919
2       1            0.832866       0.850491
3       1            0.909043       0.908579
4       0            0.807145       0.832473
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.28737353692
0.347278707709
0.0875880990649
0.0802215636904
0.0579699103278
# # # # # # # # # # 

in model: SGDClassifier  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.832084       0.837336
1       1            0.685704       0.686981
2       1            0.730518       0.731348
3       1            0.935647       0.916707
4       1            0.859949       0.871089
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.571042552862
0.690783168515
0.173900312007
0.159262386759
0.115169518071
# # # # # # # # # # 

in model: SGDClassifier  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.781120       0.800874
1       1            0.787260       0.805181
2       1            0.856362       0.861464
3       1            0.903324       0.894651
4       1            0.943478       0.926373
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.847873363794
1.02581162623
0.259006761225
0.237203075483
0.171697360227
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier
0  0            0.294930       0.282624
1  1            0.353481       0.341937
2  2            0.076102       0.086336
3  3            0.067646       0.079068
4  4            0.041055       0.057232
AUC train 0.85790559754
AUC train 0.857519048398
AUC train 0.857735856206

in model: GaussianNB  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.463085       0.482879    0.010222
1       1            0.565020       0.519919    0.955146
2       1            0.832866       0.850491    1.000000
3       1            0.909043       0.908579    1.000000
4       0            0.807145       0.832473    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
3.73382016635e-06
2.68752718603e-05
1.66664037372e-11
8.11278530554e-12
2.94716158798e-12
# # # # # # # # # # 

in model: GaussianNB  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.832084       0.837336    1.000000
1       1            0.685704       0.686981    0.999992
2       1            0.730518       0.731348    0.999999
3       1            0.935647       0.916707    1.000000
4       1            0.859949       0.871089    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
7.62014641025e-06
5.4846022659e-05
3.44179791742e-11
1.67646631477e-11
6.08521664247e-12
# # # # # # # # # # 

in model: GaussianNB  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.781120       0.800874         1.0
1       1            0.787260       0.805181         1.0
2       1            0.856362       0.861464         1.0
3       1            0.903324       0.894651         1.0
4       1            0.943478       0.926373         1.0
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
1.12800325483e-05
8.12207016265e-05
5.01936392166e-11
2.44208171368e-11
8.84010611364e-12
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB
0  0            0.294930       0.282624  3.760011e-06
1  1            0.353481       0.341937  2.707357e-05
2  2            0.076102       0.086336  1.673121e-11
3  3            0.067646       0.079068  8.140272e-12
4  4            0.041055       0.057232  2.946702e-12
AUC train 0.85800661551
AUC train 0.85778335621
AUC train 0.857959244629

in model: CV  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.463085       0.482879    0.010222  0.458479
1       1            0.565020       0.519919    0.955146  0.525615
2       1            0.832866       0.850491    1.000000  0.826022
3       1            0.909043       0.908579    1.000000  0.901932
4       0            0.807145       0.832473    1.000000  0.806887
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.308846281285
0.347373031234
0.0750531385195
0.0618432664626
0.0372106368202
# # # # # # # # # # 

in model: CV  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.832084       0.837336    1.000000  0.824204
1       1            0.685704       0.686981    0.999992  0.671342
2       1            0.730518       0.731348    0.999999  0.713419
3       1            0.935647       0.916707    1.000000  0.928271
4       1            0.859949       0.871089    1.000000  0.865308
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.614203753429
0.692793762029
0.148603087392
0.123620115488
0.0728210998907
# # # # # # # # # # 

in model: CV  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.781120       0.800874         1.0  0.807205
1       1            0.787260       0.805181         1.0  0.809934
2       1            0.856362       0.861464         1.0  0.861194
3       1            0.903324       0.894651         1.0  0.902841
4       1            0.943478       0.926373         1.0  0.953815
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.923252887613
1.03412768866
0.222670683353
0.18527928338
0.107941616571
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV
0  0            0.294930       0.282624  3.760011e-06  0.307751
1  1            0.353481       0.341937  2.707357e-05  0.344709
2  2            0.076102       0.086336  1.673121e-11  0.074224
3  3            0.067646       0.079068  8.140272e-12  0.061760
4  4            0.041055       0.057232  2.946702e-12  0.035981
AUC train 0.858002416623
AUC train 0.857777712214
AUC train 0.857955292429

in model: RandomForest  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.463085       0.482879    0.010222  0.458479   
1       1            0.565020       0.519919    0.955146  0.525615   
2       1            0.832866       0.850491    1.000000  0.826022   
3       1            0.909043       0.908579    1.000000  0.901932   
4       0            0.807145       0.832473    1.000000  0.806887   

   RandomForest  
0      0.375733  
1      0.641828  
2      0.901107  
3      0.901107  
4      0.901107  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
0.375733476133
0.375733476133
0.122935180429
0.122935180429
0.122935180429
# # # # # # # # # # 

in model: RandomForest  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.832084       0.837336    1.000000  0.824204   
1       1            0.685704       0.686981    0.999992  0.671342   
2       1            0.730518       0.731348    0.999999  0.713419   
3       1            0.935647       0.916707    1.000000  0.928271   
4       1            0.859949       0.871089    1.000000  0.865308   

   RandomForest  
0      0.901155  
1      0.642456  
2      0.642456  
3      0.901155  
4      0.901155  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
0.757275813181
0.757275813181
0.248205186871
0.248205186871
0.248205186871
# # # # # # # # # # 

in model: RandomForest  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.781120       0.800874         1.0  0.807205   
1       1            0.787260       0.805181         1.0  0.809934   
2       1            0.856362       0.861464         1.0  0.861194   
3       1            0.903324       0.894651         1.0  0.902841   
4       1            0.943478       0.926373         1.0  0.953815   

   RandomForest  
0      0.901174  
1      0.901174  
2      0.901174  
3      0.901174  
4      0.901174  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
1.13345794204
1.13345794204
0.371313771449
0.371313771449
0.371313771449
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest
0  0            0.294930       0.282624  3.760011e-06  0.307751      0.377819
1  1            0.353481       0.341937  2.707357e-05  0.344709      0.377819
2  2            0.076102       0.086336  1.673121e-11  0.074224      0.123771
3  3            0.067646       0.079068  8.140272e-12  0.061760      0.123771
4  4            0.041055       0.057232  2.946702e-12  0.035981      0.123771
AUC train 0.841537111714
AUC train 0.842976015435
AUC train 0.84267797234

in model: Neural_net  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.463085       0.482879    0.010222  0.458479   
1       1            0.565020       0.519919    0.955146  0.525615   
2       1            0.832866       0.850491    1.000000  0.826022   
3       1            0.909043       0.908579    1.000000  0.901932   
4       0            0.807145       0.832473    1.000000  0.806887   

   RandomForest  Neural_net  
0      0.375733    0.465278  
1      0.641828    0.555851  
2      0.901107    0.834059  
3      0.901107    0.909069  
4      0.901107    0.807369  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.293577476754
0.355372780597
0.0786624297397
0.0707479630574
0.0347223372043
# # # # # # # # # # 

in model: Neural_net  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.832084       0.837336    1.000000  0.824204   
1       1            0.685704       0.686981    0.999992  0.671342   
2       1            0.730518       0.731348    0.999999  0.713419   
3       1            0.935647       0.916707    1.000000  0.928271   
4       1            0.859949       0.871089    1.000000  0.865308   

   RandomForest  Neural_net  
0      0.901155    0.831338  
1      0.642456    0.689320  
2      0.642456    0.730222  
3      0.901155    0.931828  
4      0.901155    0.859209  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.591555205008
0.714514518761
0.157389814194
0.142078283382
0.0677914687967
# # # # # # # # # # 

in model: Neural_net  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.781120       0.800874         1.0  0.807205   
1       1            0.787260       0.805181         1.0  0.809934   
2       1            0.856362       0.861464         1.0  0.861194   
3       1            0.903324       0.894651         1.0  0.902841   
4       1            0.943478       0.926373         1.0  0.953815   

   RandomForest  Neural_net  
0      0.901174    0.779433  
1      0.901174    0.784678  
2      0.901174    0.854811  
3      0.901174    0.902589  
4      0.901174    0.944679  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.878935546336
1.06786560657
0.24446089954
0.220848326437
0.105815410099
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.294930       0.282624  3.760011e-06  0.307751      0.377819   
1  1            0.353481       0.341937  2.707357e-05  0.344709      0.377819   
2  2            0.076102       0.086336  1.673121e-11  0.074224      0.123771   
3  3            0.067646       0.079068  8.140272e-12  0.061760      0.123771   
4  4            0.041055       0.057232  2.946702e-12  0.035981      0.123771   

   Neural_net  
0    0.292979  
1    0.355955  
2    0.081487  
3    0.073616  
4    0.035272  
AUC train 0.857696625137
AUC train 0.857364619316
AUC train 0.857514501709

in model: DART  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857648	valid_1's auc: 0.857722
[20]	training's auc: 0.857713	valid_1's auc: 0.857745
[30]	training's auc: 0.857738	valid_1's auc: 0.857725
[40]	training's auc: 0.857749	valid_1's auc: 0.857739
Early stopping, best iteration is:
[21]	training's auc: 0.857715	valid_1's auc: 0.857748
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.463085       0.482879    0.010222  0.458479   
1       1            0.565020       0.519919    0.955146  0.525615   
2       1            0.832866       0.850491    1.000000  0.826022   
3       1            0.909043       0.908579    1.000000  0.901932   
4       0            0.807145       0.832473    1.000000  0.806887   

   RandomForest  Neural_net      DART  
0      0.375733    0.465278  0.452418  
1      0.641828    0.555851  0.531508  
2      0.901107    0.834059  0.817932  
3      0.901107    0.909069  0.893103  
4      0.901107    0.807369  0.789130  
# # # # # # # # # # 
0.298229302624
0.361071678533
0.101669849682
0.07435797169
0.0356857679639
# # # # # # # # # # 

in model: DART  k-fold: 2 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857597	valid_1's auc: 0.857431
[20]	training's auc: 0.857637	valid_1's auc: 0.857043
[30]	training's auc: 0.857659	valid_1's auc: 0.856877
Early stopping, best iteration is:
[10]	training's auc: 0.857597	valid_1's auc: 0.857431
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.832084       0.837336    1.000000  0.824204   
1       1            0.685704       0.686981    0.999992  0.671342   
2       1            0.730518       0.731348    0.999999  0.713419   
3       1            0.935647       0.916707    1.000000  0.928271   
4       1            0.859949       0.871089    1.000000  0.865308   

   RandomForest  Neural_net      DART  
0      0.901155    0.831338  0.828120  
1      0.642456    0.689320  0.651172  
2      0.642456    0.730222  0.700958  
3      0.901155    0.931828  0.939857  
4      0.901155    0.859209  0.876496  
# # # # # # # # # # 
0.591126199079
0.72436073973
0.192712203334
0.155309651633
0.0729972005482
# # # # # # # # # # 

in model: DART  k-fold: 3 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.85769	valid_1's auc: 0.856633
[20]	training's auc: 0.857738	valid_1's auc: 0.856634
Early stopping, best iteration is:
[5]	training's auc: 0.857581	valid_1's auc: 0.857623
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.781120       0.800874         1.0  0.807205   
1       1            0.787260       0.805181         1.0  0.809934   
2       1            0.856362       0.861464         1.0  0.861194   
3       1            0.903324       0.894651         1.0  0.902841   
4       1            0.943478       0.926373         1.0  0.953815   

   RandomForest  Neural_net      DART  
0      0.901174    0.779433  0.769719  
1      0.901174    0.784678  0.769719  
2      0.901174    0.854811  0.853691  
3      0.901174    0.902589  0.885573  
4      0.901174    0.944679  0.933102  
# # # # # # # # # # 
0.889373790014
1.08736384766
0.304375500039
0.225983901933
0.114488680827
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.294930       0.282624  3.760011e-06  0.307751      0.377819   
1  1            0.353481       0.341937  2.707357e-05  0.344709      0.377819   
2  2            0.076102       0.086336  1.673121e-11  0.074224      0.123771   
3  3            0.067646       0.079068  8.140272e-12  0.061760      0.123771   
4  4            0.041055       0.057232  2.946702e-12  0.035981      0.123771   

   Neural_net      DART  
0    0.292979  0.296458  
1    0.355955  0.362455  
2    0.081487  0.101459  
3    0.073616  0.075328  
4    0.035272  0.038163  
AUC train 0.857751897039
AUC train 0.857429813215
AUC train 0.857625008642

in model: GOSS  k-fold: 1 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857928	valid_1's auc: 0.858025
[20]	training's auc: 0.857919	valid_1's auc: 0.857927
[30]	training's auc: 0.8579	valid_1's auc: 0.85785
Early stopping, best iteration is:
[10]	training's auc: 0.857928	valid_1's auc: 0.858025
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.463085       0.482879    0.010222  0.458479   
1       1            0.565020       0.519919    0.955146  0.525615   
2       1            0.832866       0.850491    1.000000  0.826022   
3       1            0.909043       0.908579    1.000000  0.901932   
4       0            0.807145       0.832473    1.000000  0.806887   

   RandomForest  Neural_net      DART      GOSS  
0      0.375733    0.465278  0.452418  0.454269  
1      0.641828    0.555851  0.531508  0.517998  
2      0.901107    0.834059  0.817932  0.820134  
3      0.901107    0.909069  0.893103  0.887281  
4      0.901107    0.807369  0.789130  0.798974  
# # # # # # # # # # 
0.316694250972
0.3618579765
0.0914429705313
0.0740311506249
0.0591577306789
# # # # # # # # # # 

in model: GOSS  k-fold: 2 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858056	valid_1's auc: 0.857426
[20]	training's auc: 0.858057	valid_1's auc: 0.857265
Early stopping, best iteration is:
[7]	training's auc: 0.858048	valid_1's auc: 0.857721
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.832084       0.837336    1.000000  0.824204   
1       1            0.685704       0.686981    0.999992  0.671342   
2       1            0.730518       0.731348    0.999999  0.713419   
3       1            0.935647       0.916707    1.000000  0.928271   
4       1            0.859949       0.871089    1.000000  0.865308   

   RandomForest  Neural_net      DART      GOSS  
0      0.901155    0.831338  0.828120  0.789174  
1      0.642456    0.689320  0.651172  0.656399  
2      0.642456    0.730222  0.700958  0.696099  
3      0.901155    0.931828  0.939857  0.877378  
4      0.901155    0.859209  0.876496  0.829521  
# # # # # # # # # # 
0.649141172485
0.721736958073
0.213238648653
0.178827386567
0.143742506706
# # # # # # # # # # 

in model: GOSS  k-fold: 3 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857959	valid_1's auc: 0.857951
[20]	training's auc: 0.857956	valid_1's auc: 0.857899
Early stopping, best iteration is:
[7]	training's auc: 0.857948	valid_1's auc: 0.857954
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.781120       0.800874         1.0  0.807205   
1       1            0.787260       0.805181         1.0  0.809934   
2       1            0.856362       0.861464         1.0  0.861194   
3       1            0.903324       0.894651         1.0  0.902841   
4       1            0.943478       0.926373         1.0  0.953815   

   RandomForest  Neural_net      DART      GOSS  
0      0.901174    0.779433  0.769719  0.773564  
1      0.901174    0.784678  0.769719  0.783664  
2      0.901174    0.854811  0.853691  0.820293  
3      0.901174    0.902589  0.885573  0.860027  
4      0.901174    0.944679  0.933102  0.912665  
# # # # # # # # # # 
0.978522510748
1.07822253239
0.332622905394
0.282262724459
0.231102883765
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.294930       0.282624  3.760011e-06  0.307751      0.377819   
1  1            0.353481       0.341937  2.707357e-05  0.344709      0.377819   
2  2            0.076102       0.086336  1.673121e-11  0.074224      0.123771   
3  3            0.067646       0.079068  8.140272e-12  0.061760      0.123771   
4  4            0.041055       0.057232  2.946702e-12  0.035981      0.123771   

   Neural_net      DART      GOSS  
0    0.292979  0.296458  0.326174  
1    0.355955  0.362455  0.359408  
2    0.081487  0.101459  0.110874  
3    0.073616  0.075328  0.094088  
4    0.035272  0.038163  0.077034  
AUC train 0.858024627592
AUC train 0.857720789433
AUC train 0.857953625927

in model: LIGHT_RF  k-fold: 1 / 3

Training until validation scores don't improve for 15 rounds.
[10]	training's auc: 0.8583	valid_1's auc: 0.857986
[20]	training's auc: 0.858317	valid_1's auc: 0.858006
[30]	training's auc: 0.858326	valid_1's auc: 0.858014
[40]	training's auc: 0.85833	valid_1's auc: 0.85802
[50]	training's auc: 0.858333	valid_1's auc: 0.858023
[60]	training's auc: 0.858335	valid_1's auc: 0.858025
[70]	training's auc: 0.858336	valid_1's auc: 0.858029
[80]	training's auc: 0.858337	valid_1's auc: 0.858028
[90]	training's auc: 0.858337	valid_1's auc: 0.858029
[100]	training's auc: 0.858338	valid_1's auc: 0.858029
Early stopping, best iteration is:
[87]	training's auc: 0.858337	valid_1's auc: 0.858029
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.463085       0.482879    0.010222  0.458479   
1       1            0.565020       0.519919    0.955146  0.525615   
2       1            0.832866       0.850491    1.000000  0.826022   
3       1            0.909043       0.908579    1.000000  0.901932   
4       0            0.807145       0.832473    1.000000  0.806887   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  
0      0.375733    0.465278  0.452418  0.454269  0.457797  
1      0.641828    0.555851  0.531508  0.517998  0.528590  
2      0.901107    0.834059  0.817932  0.820134  0.785398  
3      0.901107    0.909069  0.893103  0.887281  0.833690  
4      0.901107    0.807369  0.789130  0.798974  0.776304  
# # # # # # # # # # 
0.314797795646
0.354734541279
0.158378193112
0.147226655118
0.139098146091
# # # # # # # # # # 

in model: LIGHT_RF  k-fold: 2 / 3

Training until validation scores don't improve for 15 rounds.
[10]	training's auc: 0.85843	valid_1's auc: 0.857643
[20]	training's auc: 0.858452	valid_1's auc: 0.857692
[30]	training's auc: 0.858456	valid_1's auc: 0.857709
[40]	training's auc: 0.858461	valid_1's auc: 0.857723
[50]	training's auc: 0.858463	valid_1's auc: 0.857735
[60]	training's auc: 0.858465	valid_1's auc: 0.857747
[70]	training's auc: 0.858467	valid_1's auc: 0.857751
[80]	training's auc: 0.858468	valid_1's auc: 0.857752
[90]	training's auc: 0.858468	valid_1's auc: 0.857752
[100]	training's auc: 0.858469	valid_1's auc: 0.857751
[110]	training's auc: 0.85847	valid_1's auc: 0.857755
[120]	training's auc: 0.85847	valid_1's auc: 0.857756
[130]	training's auc: 0.858471	valid_1's auc: 0.857757
[140]	training's auc: 0.858471	valid_1's auc: 0.857759
[150]	training's auc: 0.858471	valid_1's auc: 0.857761
[160]	training's auc: 0.858472	valid_1's auc: 0.857762
[170]	training's auc: 0.858472	valid_1's auc: 0.857761
[180]	training's auc: 0.858472	valid_1's auc: 0.857761
Early stopping, best iteration is:
[168]	training's auc: 0.858472	valid_1's auc: 0.857762
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.832084       0.837336    1.000000  0.824204   
1       1            0.685704       0.686981    0.999992  0.671342   
2       1            0.730518       0.731348    0.999999  0.713419   
3       1            0.935647       0.916707    1.000000  0.928271   
4       1            0.859949       0.871089    1.000000  0.865308   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  
0      0.901155    0.831338  0.828120  0.789174  0.788140  
1      0.642456    0.689320  0.651172  0.656399  0.659039  
2      0.642456    0.730222  0.700958  0.696099  0.701241  
3      0.901155    0.931828  0.939857  0.877378  0.844073  
4      0.901155    0.859209  0.876496  0.829521  0.811355  
# # # # # # # # # # 
0.632376171084
0.70979064549
0.314592756285
0.2940064746
0.273540804158
# # # # # # # # # # 

in model: LIGHT_RF  k-fold: 3 / 3

Training until validation scores don't improve for 15 rounds.
[10]	training's auc: 0.858336	valid_1's auc: 0.857927
[20]	training's auc: 0.858356	valid_1's auc: 0.857947
[30]	training's auc: 0.858361	valid_1's auc: 0.857958
[40]	training's auc: 0.858365	valid_1's auc: 0.857962
[50]	training's auc: 0.858368	valid_1's auc: 0.857969
[60]	training's auc: 0.85837	valid_1's auc: 0.85797
[70]	training's auc: 0.858372	valid_1's auc: 0.85797
[80]	training's auc: 0.858372	valid_1's auc: 0.857969
Early stopping, best iteration is:
[72]	training's auc: 0.858372	valid_1's auc: 0.857971
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.781120       0.800874         1.0  0.807205   
1       1            0.787260       0.805181         1.0  0.809934   
2       1            0.856362       0.861464         1.0  0.861194   
3       1            0.903324       0.894651         1.0  0.902841   
4       1            0.943478       0.926373         1.0  0.953815   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  
0      0.901174    0.779433  0.769719  0.773564  0.776962  
1      0.901174    0.784678  0.769719  0.783664  0.780161  
2      0.901174    0.854811  0.853691  0.820293  0.807565  
3      0.901174    0.902589  0.885573  0.860027  0.830799  
4      0.901174    0.944679  0.933102  0.912665  0.860838  
# # # # # # # # # # 
0.949188643867
1.05373267584
0.468613299937
0.441949892471
0.412653387305
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.294930       0.282624  3.760011e-06  0.307751      0.377819   
1  1            0.353481       0.341937  2.707357e-05  0.344709      0.377819   
2  2            0.076102       0.086336  1.673121e-11  0.074224      0.123771   
3  3            0.067646       0.079068  8.140272e-12  0.061760      0.123771   
4  4            0.041055       0.057232  2.946702e-12  0.035981      0.123771   

   Neural_net      DART      GOSS  LIGHT_RF  
0    0.292979  0.296458  0.326174  0.316396  
1    0.355955  0.362455  0.359408  0.351244  
2    0.081487  0.101459  0.110874  0.156204  
3    0.073616  0.075328  0.094088  0.147317  
4    0.035272  0.038163  0.077034  0.137551  
AUC train 0.858029335019
AUC train 0.857762475205
AUC train 0.857970674821

in model: LIGHTgbm  k-fold: 1 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858092	valid_1's auc: 0.857995
[20]	training's auc: 0.858231	valid_1's auc: 0.857949
Early stopping, best iteration is:
[7]	training's auc: 0.85805	valid_1's auc: 0.85801
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.463085       0.482879    0.010222  0.458479   
1       1            0.565020       0.519919    0.955146  0.525615   
2       1            0.832866       0.850491    1.000000  0.826022   
3       1            0.909043       0.908579    1.000000  0.901932   
4       0            0.807145       0.832473    1.000000  0.806887   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0      0.375733    0.465278  0.452418  0.454269  0.457797  0.449602  
1      0.641828    0.555851  0.531508  0.517998  0.528590  0.526854  
2      0.901107    0.834059  0.817932  0.820134  0.785398  0.804630  
3      0.901107    0.909069  0.893103  0.887281  0.833690  0.865644  
4      0.901107    0.807369  0.789130  0.798974  0.776304  0.793460  
# # # # # # # # # # 
0.321338891943
0.357518790514
0.111646202664
0.0983849526565
0.081507803719
# # # # # # # # # # 

in model: LIGHTgbm  k-fold: 2 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858224	valid_1's auc: 0.857627
[20]	training's auc: 0.858366	valid_1's auc: 0.856759
Early stopping, best iteration is:
[5]	training's auc: 0.858141	valid_1's auc: 0.857714
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.832084       0.837336    1.000000  0.824204   
1       1            0.685704       0.686981    0.999992  0.671342   
2       1            0.730518       0.731348    0.999999  0.713419   
3       1            0.935647       0.916707    1.000000  0.928271   
4       1            0.859949       0.871089    1.000000  0.865308   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0      0.901155    0.831338  0.828120  0.789174  0.788140  0.775579  
1      0.642456    0.689320  0.651172  0.656399  0.659039  0.639428  
2      0.642456    0.730222  0.700958  0.696099  0.701241  0.678110  
3      0.901155    0.931828  0.939857  0.877378  0.844073  0.856889  
4      0.901155    0.859209  0.876496  0.829521  0.811355  0.805865  
# # # # # # # # # # 
0.657292958689
0.730709469628
0.258032431915
0.233276511415
0.197345661527
# # # # # # # # # # 

in model: LIGHTgbm  k-fold: 3 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858125	valid_1's auc: 0.85794
[20]	training's auc: 0.858277	valid_1's auc: 0.857901
Early stopping, best iteration is:
[6]	training's auc: 0.858062	valid_1's auc: 0.857946
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.781120       0.800874         1.0  0.807205   
1       1            0.787260       0.805181         1.0  0.809934   
2       1            0.856362       0.861464         1.0  0.861194   
3       1            0.903324       0.894651         1.0  0.902841   
4       1            0.943478       0.926373         1.0  0.953815   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0      0.901174    0.779433  0.769719  0.773564  0.776962  0.780049  
1      0.901174    0.784678  0.769719  0.783664  0.780161  0.776639  
2      0.901174    0.854811  0.853691  0.820293  0.807565  0.818519  
3      0.901174    0.902589  0.885573  0.860027  0.830799  0.854150  
4      0.901174    0.944679  0.933102  0.912665  0.860838  0.899186  
# # # # # # # # # # 
0.985641189575
1.08734105644
0.381481846056
0.349302180879
0.295229437533
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.294930       0.282624  3.760011e-06  0.307751      0.377819   
1  1            0.353481       0.341937  2.707357e-05  0.344709      0.377819   
2  2            0.076102       0.086336  1.673121e-11  0.074224      0.123771   
3  3            0.067646       0.079068  8.140272e-12  0.061760      0.123771   
4  4            0.041055       0.057232  2.946702e-12  0.035981      0.123771   

   Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0    0.292979  0.296458  0.326174  0.316396  0.328547  
1    0.355955  0.362455  0.359408  0.351244  0.362447  
2    0.081487  0.101459  0.110874  0.156204  0.127161  
3    0.073616  0.075328  0.094088  0.147317  0.116434  
4    0.035272  0.038163  0.077034  0.137551  0.098410  
AUC train 0.858009934285
AUC train 0.857714191597
AUC train 0.857946374634

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of rows: 2556790
number of columns: 11

'id',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459140
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of columns: 11
number of data: 2556790
<<<<<<<<<<<<<<<<<<<<
saving DONE.
ON LEVEL: 4

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of rows: 2459140
number of columns: 11

'target',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of rows: 2556790
number of columns: 11

'id',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

in model: LogisticRegression  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.429692
1       1            0.489228
2       1            0.859614
3       1            0.924470
4       0            0.828028
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.2975654575
0.347479203895
0.067038599795
0.056803743535
0.0411083145306
# # # # # # # # # # 

in model: LogisticRegression  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.820528
1       1            0.658797
2       1            0.703082
3       1            0.924059
4       1            0.867836
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.613882822543
0.71183412094
0.144583431054
0.124804688126
0.0917895765744
# # # # # # # # # # 

in model: LogisticRegression  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.801268
1       1            0.806836
2       1            0.870020
3       1            0.907754
4       1            0.942573
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.928845469435
1.07105246
0.220564048687
0.190076438984
0.142460896373
# # # # # # # # # # 
  id  LogisticRegression
0  0            0.309615
1  1            0.357017
2  2            0.073521
3  3            0.063359
4  4            0.047487
AUC train 0.857675561664
AUC train 0.857126905473
AUC train 0.857367425155

in model: SGDClassifier  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  "and default tol will be 1e-3." % type(self), FutureWarning)
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.429692       0.471098
1       1            0.489228       0.497499
2       1            0.859614       0.853678
3       1            0.924470       0.905156
4       0            0.828028       0.833706
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.277914095002
0.337536103713
0.0845203158968
0.0763560481531
0.0639774209872
# # # # # # # # # # 

in model: SGDClassifier  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.820528       0.851279
1       1            0.658797       0.678398
2       1            0.703082       0.734327
3       1            0.924059       0.919506
4       1            0.867836       0.881698
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.562720218049
0.683368712407
0.170379631967
0.153592087728
0.128664885516
# # # # # # # # # # 

in model: SGDClassifier  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.801268       0.831721
1       1            0.806836       0.834241
2       1            0.870020       0.880820
3       1            0.907754       0.906397
4       1            0.942573       0.931645
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.858077109297
1.03965379327
0.2607970234
0.235175550395
0.197211577215
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier
0  0            0.309615       0.286026
1  1            0.357017       0.346551
2  2            0.073521       0.086932
3  3            0.063359       0.078392
4  4            0.047487       0.065737
AUC train 0.857900004708
AUC train 0.857481506556
AUC train 0.857696795871

in model: GaussianNB  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.429692       0.471098    0.008616
1       1            0.489228       0.497499    0.984002
2       1            0.859614       0.853678    1.000000
3       1            0.924470       0.905156    1.000000
4       0            0.828028       0.833706    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
2.28576202777e-06
2.80422304663e-05
1.68613146983e-11
9.23956185724e-12
2.88956870085e-12
# # # # # # # # # # 

in model: GaussianNB  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.820528       0.851279    1.000000
1       1            0.658797       0.678398    0.999992
2       1            0.703082       0.734327    0.999999
3       1            0.924059       0.919506    1.000000
4       1            0.867836       0.881698    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
4.84897135758e-06
5.88459300166e-05
3.77711408476e-11
2.07740076528e-11
6.53231082622e-12
# # # # # # # # # # 

in model: GaussianNB  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.801268       0.831721         1.0
1       1            0.806836       0.834241         1.0
2       1            0.870020       0.880820         1.0
3       1            0.907754       0.906397         1.0
4       1            0.942573       0.931645         1.0
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
7.13165149198e-06
8.65712309455e-05
5.43556235006e-11
2.98530397361e-11
9.33921288471e-12
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB
0  0            0.309615       0.286026  2.377217e-06
1  1            0.357017       0.346551  2.885708e-05
2  2            0.073521       0.086932  1.811854e-11
3  3            0.063359       0.078392  9.951013e-12
4  4            0.047487       0.065737  3.113071e-12
AUC train 0.858039550505
AUC train 0.857776529019
AUC train 0.857986242787

in model: CV  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.429692       0.471098    0.008616  0.453373
1       1            0.489228       0.497499    0.984002  0.542104
2       1            0.859614       0.853678    1.000000  0.834104
3       1            0.924470       0.905156    1.000000  0.907729
4       0            0.828028       0.833706    1.000000  0.818774
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.304395951088
0.346472706742
0.0728381925943
0.0620178200496
0.0296742185835
# # # # # # # # # # 

in model: CV  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.820528       0.851279    1.000000  0.824305
1       1            0.658797       0.678398    0.999992  0.665656
2       1            0.703082       0.734327    0.999999  0.715541
3       1            0.924059       0.919506    1.000000  0.924146
4       1            0.867836       0.881698    1.000000  0.857162
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.611531405589
0.695692962913
0.149657996074
0.126456372929
0.0663263496667
# # # # # # # # # # 

in model: CV  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.801268       0.831721         1.0  0.789376
1       1            0.806836       0.834241         1.0  0.793108
2       1            0.870020       0.880820         1.0  0.844660
3       1            0.907754       0.906397         1.0  0.895376
4       1            0.942573       0.931645         1.0  0.945358
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.921865392396
1.03668163303
0.226737462522
0.192019537372
0.0993065945834
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV
0  0            0.309615       0.286026  2.377217e-06  0.307288
1  1            0.357017       0.346551  2.885708e-05  0.345561
2  2            0.073521       0.086932  1.811854e-11  0.075579
3  3            0.063359       0.078392  9.951013e-12  0.064007
4  4            0.047487       0.065737  3.113071e-12  0.033102
AUC train 0.858029072214
AUC train 0.857771833627
AUC train 0.857977155121

in model: RandomForest  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.429692       0.471098    0.008616  0.453373   
1       1            0.489228       0.497499    0.984002  0.542104   
2       1            0.859614       0.853678    1.000000  0.834104   
3       1            0.924470       0.905156    1.000000  0.907729   
4       0            0.828028       0.833706    1.000000  0.818774   

   RandomForest  
0      0.374711  
1      0.642053  
2      0.901341  
3      0.901341  
4      0.901341  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
0.374710678612
0.374710678612
0.12205857087
0.12205857087
0.12205857087
# # # # # # # # # # 

in model: RandomForest  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.820528       0.851279    1.000000  0.824305   
1       1            0.658797       0.678398    0.999992  0.665656   
2       1            0.703082       0.734327    0.999999  0.715541   
3       1            0.924059       0.919506    1.000000  0.924146   
4       1            0.867836       0.881698    1.000000  0.857162   

   RandomForest  
0      0.899223  
1      0.637128  
2      0.637128  
3      0.899223  
4      0.899223  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
0.752370485206
0.752370485206
0.246377997499
0.246377997499
0.246377997499
# # # # # # # # # # 

in model: RandomForest  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.801268       0.831721         1.0  0.789376   
1       1            0.806836       0.834241         1.0  0.793108   
2       1            0.870020       0.880820         1.0  0.844660   
3       1            0.907754       0.906397         1.0  0.895376   
4       1            0.942573       0.931645         1.0  0.945358   

   RandomForest  
0      0.848498  
1      0.874157  
2      0.900415  
3      0.900415  
4      0.900415  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
1.12986496175
1.12986496175
0.369911944356
0.369911944356
0.369911944356
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest
0  0            0.309615       0.286026  2.377217e-06  0.307288      0.376622
1  1            0.357017       0.346551  2.885708e-05  0.345561      0.376622
2  2            0.073521       0.086932  1.811854e-11  0.075579      0.123304
3  3            0.063359       0.078392  9.951013e-12  0.064007      0.123304
4  4            0.047487       0.065737  3.113071e-12  0.033102      0.123304
AUC train 0.841285791645
AUC train 0.843503930197
AUC train 0.842224931325

in model: Neural_net  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.429692       0.471098    0.008616  0.453373   
1       1            0.489228       0.497499    0.984002  0.542104   
2       1            0.859614       0.853678    1.000000  0.834104   
3       1            0.924470       0.905156    1.000000  0.907729   
4       0            0.828028       0.833706    1.000000  0.818774   

   RandomForest  Neural_net  
0      0.374711    0.434150  
1      0.642053    0.500111  
2      0.901341    0.858835  
3      0.901341    0.921060  
4      0.901341    0.829972  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.301432856061
0.348867130661
0.0711446628048
0.0610607312227
0.0262672637435
# # # # # # # # # # 

in model: Neural_net  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.820528       0.851279    1.000000  0.824305   
1       1            0.658797       0.678398    0.999992  0.665656   
2       1            0.703082       0.734327    0.999999  0.715541   
3       1            0.924059       0.919506    1.000000  0.924146   
4       1            0.867836       0.881698    1.000000  0.857162   

   RandomForest  Neural_net  
0      0.899223    0.829867  
1      0.637128    0.659613  
2      0.637128    0.704583  
3      0.899223    0.923126  
4      0.899223    0.871488  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.615479439303
0.708568206722
0.159350199043
0.12798841303
0.0617886365216
# # # # # # # # # # 

in model: Neural_net  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.801268       0.831721         1.0  0.789376   
1       1            0.806836       0.834241         1.0  0.793108   
2       1            0.870020       0.880820         1.0  0.844660   
3       1            0.907754       0.906397         1.0  0.895376   
4       1            0.942573       0.931645         1.0  0.945358   

   RandomForest  Neural_net  
0      0.848498    0.808607  
1      0.874157    0.814151  
2      0.900415    0.870134  
3      0.900415    0.905594  
4      0.900415    0.939334  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.93350765575
1.06752978509
0.239888953892
0.196498772467
0.0970262928782
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.309615       0.286026  2.377217e-06  0.307288      0.376622   
1  1            0.357017       0.346551  2.885708e-05  0.345561      0.376622   
2  2            0.073521       0.086932  1.811854e-11  0.075579      0.123304   
3  3            0.063359       0.078392  9.951013e-12  0.064007      0.123304   
4  4            0.047487       0.065737  3.113071e-12  0.033102      0.123304   

   Neural_net  
0    0.311169  
1    0.355843  
2    0.079963  
3    0.065500  
4    0.032342  
AUC train 0.85773628243
AUC train 0.857325585384
AUC train 0.85747428503

in model: DART  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857422	valid_1's auc: 0.857461
[20]	training's auc: 0.857462	valid_1's auc: 0.857134
[30]	training's auc: 0.857463	valid_1's auc: 0.856921
Early stopping, best iteration is:
[15]	training's auc: 0.857456	valid_1's auc: 0.857492
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.429692       0.471098    0.008616  0.453373   
1       1            0.489228       0.497499    0.984002  0.542104   
2       1            0.859614       0.853678    1.000000  0.834104   
3       1            0.924470       0.905156    1.000000  0.907729   
4       0            0.828028       0.833706    1.000000  0.818774   

   RandomForest  Neural_net      DART  
0      0.374711    0.434150  0.449570  
1      0.642053    0.500111  0.552212  
2      0.901341    0.858835  0.817348  
3      0.901341    0.921060  0.925193  
4      0.901341    0.829972  0.806753  
# # # # # # # # # # 
0.300849541872
0.374356122904
0.100733090302
0.0539439623625
0.0364446640294
# # # # # # # # # # 

in model: DART  k-fold: 2 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857459	valid_1's auc: 0.85722
[20]	training's auc: 0.857504	valid_1's auc: 0.857206
Early stopping, best iteration is:
[8]	training's auc: 0.857433	valid_1's auc: 0.857233
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.820528       0.851279    1.000000  0.824305   
1       1            0.658797       0.678398    0.999992  0.665656   
2       1            0.703082       0.734327    0.999999  0.715541   
3       1            0.924059       0.919506    1.000000  0.924146   
4       1            0.867836       0.881698    1.000000  0.857162   

   RandomForest  Neural_net      DART  
0      0.899223    0.829867  0.809867  
1      0.637128    0.659613  0.658836  
2      0.637128    0.704583  0.709926  
3      0.899223    0.923126  0.906931  
4      0.899223    0.871488  0.875798  
# # # # # # # # # # 
0.598828904777
0.746546059548
0.192160327064
0.12637027233
0.0720151421604
# # # # # # # # # # 

in model: DART  k-fold: 3 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.85744	valid_1's auc: 0.857337
[20]	training's auc: 0.85747	valid_1's auc: 0.857261
[30]	training's auc: 0.857492	valid_1's auc: 0.857276
Early stopping, best iteration is:
[14]	training's auc: 0.857464	valid_1's auc: 0.857346
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.801268       0.831721         1.0  0.789376   
1       1            0.806836       0.834241         1.0  0.793108   
2       1            0.870020       0.880820         1.0  0.844660   
3       1            0.907754       0.906397         1.0  0.895376   
4       1            0.942573       0.931645         1.0  0.945358   

   RandomForest  Neural_net      DART  
0      0.848498    0.808607  0.808860  
1      0.874157    0.814151  0.808860  
2      0.900415    0.870134  0.844840  
3      0.900415    0.905594  0.877457  
4      0.900415    0.939334  0.986239  
# # # # # # # # # # 
0.896124540114
1.1149672126
0.29072543709
0.186501261366
0.107496639787
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.309615       0.286026  2.377217e-06  0.307288      0.376622   
1  1            0.357017       0.346551  2.885708e-05  0.345561      0.376622   
2  2            0.073521       0.086932  1.811854e-11  0.075579      0.123304   
3  3            0.063359       0.078392  9.951013e-12  0.064007      0.123304   
4  4            0.047487       0.065737  3.113071e-12  0.033102      0.123304   

   Neural_net      DART  
0    0.311169  0.298708  
1    0.355843  0.371656  
2    0.079963  0.096908  
3    0.065500  0.062167  
4    0.032342  0.035832  
AUC train 0.857492868798
AUC train 0.857237405475
AUC train 0.85734205501

in model: GOSS  k-fold: 1 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857917	valid_1's auc: 0.857996
[20]	training's auc: 0.857915	valid_1's auc: 0.857585
Early stopping, best iteration is:
[8]	training's auc: 0.857911	valid_1's auc: 0.858015
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.429692       0.471098    0.008616  0.453373   
1       1            0.489228       0.497499    0.984002  0.542104   
2       1            0.859614       0.853678    1.000000  0.834104   
3       1            0.924470       0.905156    1.000000  0.907729   
4       0            0.828028       0.833706    1.000000  0.818774   

   RandomForest  Neural_net      DART      GOSS  
0      0.374711    0.434150  0.449570  0.464639  
1      0.642053    0.500111  0.552212  0.525643  
2      0.901341    0.858835  0.817348  0.812910  
3      0.901341    0.921060  0.925193  0.890747  
4      0.901341    0.829972  0.806753  0.800671  
# # # # # # # # # # 
0.313160274618
0.355799816348
0.105239093453
0.0943845185021
0.0689751721145
# # # # # # # # # # 

in model: GOSS  k-fold: 2 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858038	valid_1's auc: 0.85771
[20]	training's auc: 0.858038	valid_1's auc: 0.856891
Early stopping, best iteration is:
[7]	training's auc: 0.858033	valid_1's auc: 0.857734
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.820528       0.851279    1.000000  0.824305   
1       1            0.658797       0.678398    0.999992  0.665656   
2       1            0.703082       0.734327    0.999999  0.715541   
3       1            0.924059       0.919506    1.000000  0.924146   
4       1            0.867836       0.881698    1.000000  0.857162   

   RandomForest  Neural_net      DART      GOSS  
0      0.899223    0.829867  0.809867  0.797611  
1      0.637128    0.659613  0.658836  0.656257  
2      0.637128    0.704583  0.709926  0.700340  
3      0.899223    0.923126  0.906931  0.888250  
4      0.899223    0.871488  0.875798  0.831363  
# # # # # # # # # # 
0.63354089924
0.716031869529
0.225921164121
0.200564297296
0.149423950668
# # # # # # # # # # 

in model: GOSS  k-fold: 3 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857941	valid_1's auc: 0.857917
[20]	training's auc: 0.857944	valid_1's auc: 0.857762
Early stopping, best iteration is:
[8]	training's auc: 0.857934	valid_1's auc: 0.857934
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.801268       0.831721         1.0  0.789376   
1       1            0.806836       0.834241         1.0  0.793108   
2       1            0.870020       0.880820         1.0  0.844660   
3       1            0.907754       0.906397         1.0  0.895376   
4       1            0.942573       0.931645         1.0  0.945358   

   RandomForest  Neural_net      DART      GOSS  
0      0.848498    0.808607  0.808860  0.779901  
1      0.874157    0.814151  0.808860  0.779901  
2      0.900415    0.870134  0.844840  0.839425  
3      0.900415    0.905594  0.877457  0.869345  
4      0.900415    0.939334  0.986239  0.923584  
# # # # # # # # # # 
0.959324175918
1.06209706155
0.332592064807
0.293975776894
0.221607904706
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.309615       0.286026  2.377217e-06  0.307288      0.376622   
1  1            0.357017       0.346551  2.885708e-05  0.345561      0.376622   
2  2            0.073521       0.086932  1.811854e-11  0.075579      0.123304   
3  3            0.063359       0.078392  9.951013e-12  0.064007      0.123304   
4  4            0.047487       0.065737  3.113071e-12  0.033102      0.123304   

   Neural_net      DART      GOSS  
0    0.311169  0.298708  0.319775  
1    0.355843  0.371656  0.354032  
2    0.079963  0.096908  0.110864  
3    0.065500  0.062167  0.097992  
4    0.032342  0.035832  0.073869  
AUC train 0.858014961501
AUC train 0.857734410934
AUC train 0.857934461757

in model: LIGHT_RF  k-fold: 1 / 3

Training until validation scores don't improve for 15 rounds.
[10]	training's auc: 0.858231	valid_1's auc: 0.85789
[20]	training's auc: 0.858251	valid_1's auc: 0.857938
[30]	training's auc: 0.858256	valid_1's auc: 0.857966
[40]	training's auc: 0.85826	valid_1's auc: 0.857971
[50]	training's auc: 0.858263	valid_1's auc: 0.857976
[60]	training's auc: 0.858264	valid_1's auc: 0.857979
[70]	training's auc: 0.858266	valid_1's auc: 0.857981
[80]	training's auc: 0.858266	valid_1's auc: 0.85798
[90]	training's auc: 0.858267	valid_1's auc: 0.85798
Early stopping, best iteration is:
[76]	training's auc: 0.858266	valid_1's auc: 0.857983
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.429692       0.471098    0.008616  0.453373   
1       1            0.489228       0.497499    0.984002  0.542104   
2       1            0.859614       0.853678    1.000000  0.834104   
3       1            0.924470       0.905156    1.000000  0.907729   
4       0            0.828028       0.833706    1.000000  0.818774   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  
0      0.374711    0.434150  0.449570  0.464639  0.456471  
1      0.642053    0.500111  0.552212  0.525643  0.523065  
2      0.901341    0.858835  0.817348  0.812910  0.794591  
3      0.901341    0.921060  0.925193  0.890747  0.837037  
4      0.901341    0.829972  0.806753  0.800671  0.783174  
# # # # # # # # # # 
0.313148797886
0.351506795395
0.157040174917
0.147095823033
0.133314075102
# # # # # # # # # # 

in model: LIGHT_RF  k-fold: 2 / 3

Training until validation scores don't improve for 15 rounds.
[10]	training's auc: 0.858355	valid_1's auc: 0.857585
[20]	training's auc: 0.858373	valid_1's auc: 0.857632
[30]	training's auc: 0.858378	valid_1's auc: 0.857683
[40]	training's auc: 0.858383	valid_1's auc: 0.857698
[50]	training's auc: 0.858387	valid_1's auc: 0.857698
Early stopping, best iteration is:
[43]	training's auc: 0.858384	valid_1's auc: 0.857703
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.820528       0.851279    1.000000  0.824305   
1       1            0.658797       0.678398    0.999992  0.665656   
2       1            0.703082       0.734327    0.999999  0.715541   
3       1            0.924059       0.919506    1.000000  0.924146   
4       1            0.867836       0.881698    1.000000  0.857162   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  
0      0.899223    0.829867  0.809867  0.797611  0.786928  
1      0.637128    0.659613  0.658836  0.656257  0.666417  
2      0.637128    0.704583  0.709926  0.700340  0.701330  
3      0.899223    0.923126  0.906931  0.888250  0.847636  
4      0.899223    0.871488  0.875798  0.831363  0.810278  
# # # # # # # # # # 
0.629492310539
0.702020908541
0.319933363123
0.296473097493
0.269071609722
# # # # # # # # # # 

in model: LIGHT_RF  k-fold: 3 / 3

Training until validation scores don't improve for 15 rounds.
[10]	training's auc: 0.858263	valid_1's auc: 0.857874
[20]	training's auc: 0.858276	valid_1's auc: 0.857902
[30]	training's auc: 0.858282	valid_1's auc: 0.857906
[40]	training's auc: 0.858285	valid_1's auc: 0.857906
[50]	training's auc: 0.858287	valid_1's auc: 0.857909
[60]	training's auc: 0.858288	valid_1's auc: 0.85791
[70]	training's auc: 0.85829	valid_1's auc: 0.857909
Early stopping, best iteration is:
[57]	training's auc: 0.858288	valid_1's auc: 0.857914
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.801268       0.831721         1.0  0.789376   
1       1            0.806836       0.834241         1.0  0.793108   
2       1            0.870020       0.880820         1.0  0.844660   
3       1            0.907754       0.906397         1.0  0.895376   
4       1            0.942573       0.931645         1.0  0.945358   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  
0      0.848498    0.808607  0.808860  0.779901  0.775387  
1      0.874157    0.814151  0.808860  0.779901  0.774488  
2      0.900415    0.870134  0.844840  0.839425  0.807637  
3      0.900415    0.905594  0.877457  0.869345  0.830698  
4      0.900415    0.939334  0.986239  0.923584  0.860095  
# # # # # # # # # # 
0.942743832444
1.04543963569
0.475120596964
0.444651099173
0.407222340704
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.309615       0.286026  2.377217e-06  0.307288      0.376622   
1  1            0.357017       0.346551  2.885708e-05  0.345561      0.376622   
2  2            0.073521       0.086932  1.811854e-11  0.075579      0.123304   
3  3            0.063359       0.078392  9.951013e-12  0.064007      0.123304   
4  4            0.047487       0.065737  3.113071e-12  0.033102      0.123304   

   Neural_net      DART      GOSS  LIGHT_RF  
0    0.311169  0.298708  0.319775  0.314248  
1    0.355843  0.371656  0.354032  0.348480  
2    0.079963  0.096908  0.110864  0.158374  
3    0.065500  0.062167  0.097992  0.148217  
4    0.032342  0.035832  0.073869  0.135741  
AUC train 0.857982627041
AUC train 0.85770339172
AUC train 0.857913603868

in model: LIGHTgbm  k-fold: 1 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858046	valid_1's auc: 0.857835
[20]	training's auc: 0.858164	valid_1's auc: 0.857601
Early stopping, best iteration is:
[5]	training's auc: 0.85798	valid_1's auc: 0.857957
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.429692       0.471098    0.008616  0.453373   
1       1            0.489228       0.497499    0.984002  0.542104   
2       1            0.859614       0.853678    1.000000  0.834104   
3       1            0.924470       0.905156    1.000000  0.907729   
4       0            0.828028       0.833706    1.000000  0.818774   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0      0.374711    0.434150  0.449570  0.464639  0.456471  0.470606  
1      0.642053    0.500111  0.552212  0.525643  0.523065  0.524197  
2      0.901341    0.858835  0.817348  0.812910  0.794591  0.784558  
3      0.901341    0.921060  0.925193  0.890747  0.837037  0.838163  
4      0.901341    0.829972  0.806753  0.800671  0.783174  0.769576  
# # # # # # # # # # 
0.334499361606
0.37676278579
0.149672541267
0.137499693089
0.108052076076
# # # # # # # # # # 

in model: LIGHTgbm  k-fold: 2 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858182	valid_1's auc: 0.857511
[20]	training's auc: 0.858301	valid_1's auc: 0.856945
Early stopping, best iteration is:
[8]	training's auc: 0.858154	valid_1's auc: 0.857546
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.820528       0.851279    1.000000  0.824305   
1       1            0.658797       0.678398    0.999992  0.665656   
2       1            0.703082       0.734327    0.999999  0.715541   
3       1            0.924059       0.919506    1.000000  0.924146   
4       1            0.867836       0.881698    1.000000  0.857162   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0      0.899223    0.829867  0.809867  0.797611  0.786928  0.807823  
1      0.637128    0.659613  0.658836  0.656257  0.666417  0.654083  
2      0.637128    0.704583  0.709926  0.700340  0.701330  0.699852  
3      0.899223    0.923126  0.906931  0.888250  0.847636  0.909402  
4      0.899223    0.871488  0.875798  0.831363  0.810278  0.836050  
# # # # # # # # # # 
0.652797453961
0.736503817916
0.258342523151
0.225907114393
0.174720835603
# # # # # # # # # # 

in model: LIGHTgbm  k-fold: 3 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858074	valid_1's auc: 0.857881
[20]	training's auc: 0.858186	valid_1's auc: 0.857726
Early stopping, best iteration is:
[4]	training's auc: 0.857991	valid_1's auc: 0.857916
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.801268       0.831721         1.0  0.789376   
1       1            0.806836       0.834241         1.0  0.793108   
2       1            0.870020       0.880820         1.0  0.844660   
3       1            0.907754       0.906397         1.0  0.895376   
4       1            0.942573       0.931645         1.0  0.945358   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0      0.848498    0.808607  0.808860  0.779901  0.775387  0.740309  
1      0.874157    0.814151  0.808860  0.779901  0.774488  0.740309  
2      0.900415    0.870134  0.844840  0.839425  0.807637  0.774827  
3      0.900415    0.905594  0.877457  0.869345  0.830698  0.803961  
4      0.900415    0.939334  0.986239  0.923584  0.860095  0.842383  
# # # # # # # # # # 
1.00029021532
1.10804449339
0.436335097316
0.394313077439
0.323994085155
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.309615       0.286026  2.377217e-06  0.307288      0.376622   
1  1            0.357017       0.346551  2.885708e-05  0.345561      0.376622   
2  2            0.073521       0.086932  1.811854e-11  0.075579      0.123304   
3  3            0.063359       0.078392  9.951013e-12  0.064007      0.123304   
4  4            0.047487       0.065737  3.113071e-12  0.033102      0.123304   

   Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0    0.311169  0.298708  0.319775  0.314248  0.333430  
1    0.355843  0.371656  0.354032  0.348480  0.369348  
2    0.079963  0.096908  0.110864  0.158374  0.145445  
3    0.065500  0.062167  0.097992  0.148217  0.131438  
4    0.032342  0.035832  0.073869  0.135741  0.107998  
AUC train 0.857957162694
AUC train 0.8575462124
AUC train 0.857916079211

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of rows: 2556790
number of columns: 11

'id',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459140
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of columns: 11
number of data: 2556790
<<<<<<<<<<<<<<<<<<<<
saving DONE.
ON LEVEL: 5

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of rows: 2459140
number of columns: 11

'target',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of rows: 2556790
number of columns: 11

'id',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

in model: LogisticRegression  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.479145
1       1            0.584801
2       1            0.855217
3       1            0.931112
4       0            0.844962
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.310333170251
0.349785505223
0.0717115466693
0.0602436855414
0.0389702651379
# # # # # # # # # # 

in model: LogisticRegression  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.803331
1       1            0.635432
2       1            0.705341
3       1            0.902340
4       1            0.847410
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.633933634018
0.709669499486
0.154786167261
0.132049113007
0.0851481396887
# # # # # # # # # # 

in model: LogisticRegression  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.750734
1       1            0.753294
2       1            0.819101
3       1            0.881146
4       1            0.932505
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.943008404639
1.05383316958
0.228132198055
0.19633025141
0.130742450882
# # # # # # # # # # 
  id  LogisticRegression
0  0            0.314336
1  1            0.351278
2  2            0.076044
3  3            0.065443
4  4            0.043581
AUC train 0.857897432165
AUC train 0.856497814353
AUC train 0.857608446206

in model: SGDClassifier  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
/usr/local/lib/python3.5/dist-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  "and default tol will be 1e-3." % type(self), FutureWarning)
   target  LogisticRegression  SGDClassifier
0       1            0.479145       0.471692
1       1            0.584801       0.511808
2       1            0.855217       0.856461
3       1            0.931112       0.913982
4       0            0.844962       0.844443
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.28316740785
0.341268956415
0.0850540499327
0.0749922988208
0.0617144218635
# # # # # # # # # # 

in model: SGDClassifier  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.803331       0.848471
1       1            0.635432       0.677809
2       1            0.705341       0.739241
3       1            0.902340       0.917658
4       1            0.847410       0.882805
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.579602382687
0.69865848817
0.175555381917
0.154788639405
0.127628219481
# # # # # # # # # # 

in model: SGDClassifier  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.750734       0.801501
1       1            0.753294       0.802838
2       1            0.819101       0.847158
3       1            0.881146       0.878914
4       1            0.932505       0.920665
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.861047889759
1.03563672546
0.262488002828
0.231685099249
0.190905368935
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier
0  0            0.314336       0.287016
1  1            0.351278       0.345212
2  2            0.076044       0.087496
3  3            0.065443       0.077228
4  4            0.043581       0.063635
AUC train 0.857942061714
AUC train 0.857493430939
AUC train 0.857687536205

in model: GaussianNB  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.479145       0.471692    0.005604
1       1            0.584801       0.511808    0.965718
2       1            0.855217       0.856461    1.000000
3       1            0.931112       0.913982    1.000000
4       0            0.844962       0.844443    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
2.45438274283e-06
2.76174414944e-05
1.35439348918e-11
6.61040179517e-12
2.19184728572e-12
# # # # # # # # # # 

in model: GaussianNB  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.803331       0.848471    1.000000
1       1            0.635432       0.677809    0.999989
2       1            0.705341       0.739241    0.999999
3       1            0.902340       0.917658    1.000000
4       1            0.847410       0.882805    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
5.66437892175e-06
6.20333618478e-05
3.64163925603e-11
1.79829079141e-11
6.07169228151e-12
# # # # # # # # # # 

in model: GaussianNB  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.750734       0.801501         1.0
1       1            0.753294       0.802838         1.0
2       1            0.819101       0.847158         1.0
3       1            0.881146       0.878914         1.0
4       1            0.932505       0.920665         1.0
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
8.88485177073e-06
9.6425310457e-05
5.96171315972e-11
2.94198489579e-11
9.95901436842e-12
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB
0  0            0.314336       0.287016  2.961617e-06
1  1            0.351278       0.345212  3.214177e-05
2  2            0.076044       0.087496  1.987238e-11
3  3            0.065443       0.077228  9.806616e-12
4  4            0.043581       0.063635  3.319671e-12
AUC train 0.858015307516
AUC train 0.857775591592
AUC train 0.857964471352

in model: CV  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.479145       0.471692    0.005604  0.440399
1       1            0.584801       0.511808    0.965718  0.531931
2       1            0.855217       0.856461    1.000000  0.838500
3       1            0.931112       0.913982    1.000000  0.916791
4       0            0.844962       0.844443    1.000000  0.818802
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.303204630292
0.339781059845
0.0712042139603
0.0613820521158
0.0287884192442
# # # # # # # # # # 

in model: CV  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.803331       0.848471    1.000000  0.821575
1       1            0.635432       0.677809    0.999989  0.660587
2       1            0.705341       0.739241    0.999999  0.706484
3       1            0.902340       0.917658    1.000000  0.927674
4       1            0.847410       0.882805    1.000000  0.865279
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.614746090325
0.694622356713
0.145019962981
0.126548218022
0.0622751791832
# # # # # # # # # # 

in model: CV  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.750734       0.801501         1.0  0.783424
1       1            0.753294       0.802838         1.0  0.785774
2       1            0.819101       0.847158         1.0  0.836135
3       1            0.881146       0.878914         1.0  0.880541
4       1            0.932505       0.920665         1.0  0.941144
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.930589895838
1.04466446455
0.223062419538
0.198135279554
0.103549188558
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV
0  0            0.314336       0.287016  2.961617e-06  0.310197
1  1            0.351278       0.345212  3.214177e-05  0.348221
2  2            0.076044       0.087496  1.987238e-11  0.074354
3  3            0.065443       0.077228  9.806616e-12  0.066045
4  4            0.043581       0.063635  3.319671e-12  0.034516
AUC train 0.857992168951
AUC train 0.857764551027
AUC train 0.85795188305

in model: RandomForest  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.479145       0.471692    0.005604  0.440399   
1       1            0.584801       0.511808    0.965718  0.531931   
2       1            0.855217       0.856461    1.000000  0.838500   
3       1            0.931112       0.913982    1.000000  0.916791   
4       0            0.844962       0.844443    1.000000  0.818802   

   RandomForest  
0      0.375160  
1      0.642104  
2      0.901296  
3      0.901296  
4      0.901296  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
0.375160205094
0.375160205094
0.122208310273
0.122208310273
0.122208310273
# # # # # # # # # # 

in model: RandomForest  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.803331       0.848471    1.000000  0.821575   
1       1            0.635432       0.677809    0.999989  0.660587   
2       1            0.705341       0.739241    0.999999  0.706484   
3       1            0.902340       0.917658    1.000000  0.927674   
4       1            0.847410       0.882805    1.000000  0.865279   

   RandomForest  
0      0.899436  
1      0.636157  
2      0.636157  
3      0.899436  
4      0.899436  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
0.750020028346
0.750020028346
0.245231274243
0.245231274243
0.245231274243
# # # # # # # # # # 

in model: RandomForest  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.750734       0.801501         1.0  0.783424   
1       1            0.753294       0.802838         1.0  0.785774   
2       1            0.819101       0.847158         1.0  0.836135   
3       1            0.881146       0.878914         1.0  0.880541   
4       1            0.932505       0.920665         1.0  0.941144   

   RandomForest  
0      0.901261  
1      0.901261  
2      0.901261  
3      0.901261  
4      0.901261  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
1.12588233959
1.12588233959
0.367745226182
0.367745226182
0.367745226182
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest
0  0            0.314336       0.287016  2.961617e-06  0.310197      0.375294
1  1            0.351278       0.345212  3.214177e-05  0.348221      0.375294
2  2            0.076044       0.087496  1.987238e-11  0.074354      0.122582
3  3            0.065443       0.077228  9.806616e-12  0.066045      0.122582
4  4            0.043581       0.063635  3.319671e-12  0.034516      0.122582
AUC train 0.841710070556
AUC train 0.842513498743
AUC train 0.842974873698

in model: Neural_net  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.479145       0.471692    0.005604  0.440399   
1       1            0.584801       0.511808    0.965718  0.531931   
2       1            0.855217       0.856461    1.000000  0.838500   
3       1            0.931112       0.913982    1.000000  0.916791   
4       0            0.844962       0.844443    1.000000  0.818802   

   RandomForest  Neural_net  
0      0.375160    0.479597  
1      0.642104    0.572788  
2      0.901296    0.856754  
3      0.901296    0.929073  
4      0.901296    0.846073  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.311326486471
0.348720325522
0.0739746181912
0.0641553417983
0.0264424298956
# # # # # # # # # # 

in model: Neural_net  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.803331       0.848471    1.000000  0.821575   
1       1            0.635432       0.677809    0.999989  0.660587   
2       1            0.705341       0.739241    0.999999  0.706484   
3       1            0.902340       0.917658    1.000000  0.927674   
4       1            0.847410       0.882805    1.000000  0.865279   

   RandomForest  Neural_net  
0      0.899436    0.796619  
1      0.636157    0.643246  
2      0.636157    0.706532  
3      0.899436    0.898413  
4      0.899436    0.842874  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.639182046067
0.712099738211
0.160660738635
0.141911829646
0.0678593427574
# # # # # # # # # # 

in model: Neural_net  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.750734       0.801501         1.0  0.783424   
1       1            0.753294       0.802838         1.0  0.785774   
2       1            0.819101       0.847158         1.0  0.836135   
3       1            0.881146       0.878914         1.0  0.880541   
4       1            0.932505       0.920665         1.0  0.941144   

   RandomForest  Neural_net  
0      0.901261    0.759293  
1      0.901261    0.761090  
2      0.901261    0.827271  
3      0.901261    0.883089  
4      0.901261    0.933076  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.947441816411
1.05438963119
0.23605106348
0.208291483473
0.10641315989
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.314336       0.287016  2.961617e-06  0.310197      0.375294   
1  1            0.351278       0.345212  3.214177e-05  0.348221      0.375294   
2  2            0.076044       0.087496  1.987238e-11  0.074354      0.122582   
3  3            0.065443       0.077228  9.806616e-12  0.066045      0.122582   
4  4            0.043581       0.063635  3.319671e-12  0.034516      0.122582   

   Neural_net  
0    0.315814  
1    0.351463  
2    0.078684  
3    0.069430  
4    0.035471  
AUC train 0.857926294906
AUC train 0.856759801276
AUC train 0.857684563254

in model: DART  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857484	valid_1's auc: 0.857406
[20]	training's auc: 0.857529	valid_1's auc: 0.856811
[30]	training's auc: 0.85755	valid_1's auc: 0.856744
Early stopping, best iteration is:
[13]	training's auc: 0.857505	valid_1's auc: 0.857459
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.479145       0.471692    0.005604  0.440399   
1       1            0.584801       0.511808    0.965718  0.531931   
2       1            0.855217       0.856461    1.000000  0.838500   
3       1            0.931112       0.913982    1.000000  0.916791   
4       0            0.844962       0.844443    1.000000  0.818802   

   RandomForest  Neural_net      DART  
0      0.375160    0.479597  0.439121  
1      0.642104    0.572788  0.555633  
2      0.901296    0.856754  0.848253  
3      0.901296    0.929073  0.930780  
4      0.901296    0.846073  0.813970  
# # # # # # # # # # 
0.299169961307
0.374582043375
0.080213805832
0.048559566971
0.0378230273822
# # # # # # # # # # 

in model: DART  k-fold: 2 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857662	valid_1's auc: 0.856317
[20]	training's auc: 0.857724	valid_1's auc: 0.856213
Early stopping, best iteration is:
[8]	training's auc: 0.857651	valid_1's auc: 0.85737
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.803331       0.848471    1.000000  0.821575   
1       1            0.635432       0.677809    0.999989  0.660587   
2       1            0.705341       0.739241    0.999999  0.706484   
3       1            0.902340       0.917658    1.000000  0.927674   
4       1            0.847410       0.882805    1.000000  0.865279   

   RandomForest  Neural_net      DART  
0      0.899436    0.796619  0.839949  
1      0.636157    0.643246  0.639240  
2      0.636157    0.706532  0.717342  
3      0.899436    0.898413  0.941636  
4      0.899436    0.842874  0.897243  
# # # # # # # # # # 
0.599807511294
0.74551112065
0.165847130196
0.11451939082
0.070764092562
# # # # # # # # # # 

in model: DART  k-fold: 3 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857661	valid_1's auc: 0.857461
[20]	training's auc: 0.857684	valid_1's auc: 0.857328
Early stopping, best iteration is:
[4]	training's auc: 0.857539	valid_1's auc: 0.857501
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.750734       0.801501         1.0  0.783424   
1       1            0.753294       0.802838         1.0  0.785774   
2       1            0.819101       0.847158         1.0  0.836135   
3       1            0.881146       0.878914         1.0  0.880541   
4       1            0.932505       0.920665         1.0  0.941144   

   RandomForest  Neural_net      DART  
0      0.901261    0.759293  0.785516  
1      0.901261    0.761090  0.785516  
2      0.901261    0.827271  0.828132  
3      0.901261    0.883089  0.855149  
4      0.901261    0.933076  0.900172  
# # # # # # # # # # 
0.908845877841
1.08421270937
0.275832806349
0.216998397771
0.114748731596
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.314336       0.287016  2.961617e-06  0.310197      0.375294   
1  1            0.351278       0.345212  3.214177e-05  0.348221      0.375294   
2  2            0.076044       0.087496  1.987238e-11  0.074354      0.122582   
3  3            0.065443       0.077228  9.806616e-12  0.066045      0.122582   
4  4            0.043581       0.063635  3.319671e-12  0.034516      0.122582   

   Neural_net      DART  
0    0.315814  0.302949  
1    0.351463  0.361404  
2    0.078684  0.091944  
3    0.069430  0.072333  
4    0.035471  0.038250  
AUC train 0.857459954931
AUC train 0.857404764469
AUC train 0.857501608604

in model: GOSS  k-fold: 1 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857901	valid_1's auc: 0.857537
[20]	training's auc: 0.857903	valid_1's auc: 0.856937
Early stopping, best iteration is:
[5]	training's auc: 0.857879	valid_1's auc: 0.857945
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.479145       0.471692    0.005604  0.440399   
1       1            0.584801       0.511808    0.965718  0.531931   
2       1            0.855217       0.856461    1.000000  0.838500   
3       1            0.931112       0.913982    1.000000  0.916791   
4       0            0.844962       0.844443    1.000000  0.818802   

   RandomForest  Neural_net      DART      GOSS  
0      0.375160    0.479597  0.439121  0.461384  
1      0.642104    0.572788  0.555633  0.534734  
2      0.901296    0.856754  0.848253  0.768805  
3      0.901296    0.929073  0.930780  0.840732  
4      0.901296    0.846073  0.813970  0.761217  
# # # # # # # # # # 
0.33631684124
0.368229030277
0.158538093949
0.147162748012
0.119797143356
# # # # # # # # # # 

in model: GOSS  k-fold: 2 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858015	valid_1's auc: 0.857393
[20]	training's auc: 0.858022	valid_1's auc: 0.856377
Early stopping, best iteration is:
[6]	training's auc: 0.857996	valid_1's auc: 0.857694
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.803331       0.848471    1.000000  0.821575   
1       1            0.635432       0.677809    0.999989  0.660587   
2       1            0.705341       0.739241    0.999999  0.706484   
3       1            0.902340       0.917658    1.000000  0.927674   
4       1            0.847410       0.882805    1.000000  0.865279   

   RandomForest  Neural_net      DART      GOSS  
0      0.899436    0.796619  0.839949  0.776583  
1      0.636157    0.643246  0.639240  0.634272  
2      0.636157    0.706532  0.717342  0.685502  
3      0.899436    0.898413  0.941636  0.864298  
4      0.899436    0.842874  0.897243  0.808239  
# # # # # # # # # # 
0.668528695591
0.742231513355
0.297636085776
0.266483363135
0.221481961427
# # # # # # # # # # 

in model: GOSS  k-fold: 3 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857922	valid_1's auc: 0.857853
[20]	training's auc: 0.857923	valid_1's auc: 0.85767
Early stopping, best iteration is:
[7]	training's auc: 0.857908	valid_1's auc: 0.857915
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.750734       0.801501         1.0  0.783424   
1       1            0.753294       0.802838         1.0  0.785774   
2       1            0.819101       0.847158         1.0  0.836135   
3       1            0.881146       0.878914         1.0  0.880541   
4       1            0.932505       0.920665         1.0  0.941144   

   RandomForest  Neural_net      DART      GOSS  
0      0.901261    0.759293  0.785516  0.766816  
1      0.901261    0.761090  0.785516  0.771626  
2      0.901261    0.827271  0.828132  0.815683  
3      0.901261    0.883089  0.855149  0.854510  
4      0.901261    0.933076  0.900172  0.898331  
# # # # # # # # # # 
0.993233497937
1.09685524817
0.416056290767
0.373667161122
0.303019291375
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.314336       0.287016  2.961617e-06  0.310197      0.375294   
1  1            0.351278       0.345212  3.214177e-05  0.348221      0.375294   
2  2            0.076044       0.087496  1.987238e-11  0.074354      0.122582   
3  3            0.065443       0.077228  9.806616e-12  0.066045      0.122582   
4  4            0.043581       0.063635  3.319671e-12  0.034516      0.122582   

   Neural_net      DART      GOSS  
0    0.315814  0.302949  0.331078  
1    0.351463  0.361404  0.365618  
2    0.078684  0.091944  0.138685  
3    0.069430  0.072333  0.124556  
4    0.035471  0.038250  0.101006  
AUC train 0.857944641671
AUC train 0.857693664427
AUC train 0.857915484946

in model: LIGHT_RF  k-fold: 1 / 3

Training until validation scores don't improve for 15 rounds.
[10]	training's auc: 0.858207	valid_1's auc: 0.857436
[20]	training's auc: 0.858224	valid_1's auc: 0.857457
[30]	training's auc: 0.858226	valid_1's auc: 0.857527
[40]	training's auc: 0.85823	valid_1's auc: 0.857575
[50]	training's auc: 0.858234	valid_1's auc: 0.857547
Early stopping, best iteration is:
[40]	training's auc: 0.85823	valid_1's auc: 0.857575
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.479145       0.471692    0.005604  0.440399   
1       1            0.584801       0.511808    0.965718  0.531931   
2       1            0.855217       0.856461    1.000000  0.838500   
3       1            0.931112       0.913982    1.000000  0.916791   
4       0            0.844962       0.844443    1.000000  0.818802   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  
0      0.375160    0.479597  0.439121  0.461384  0.445780  
1      0.642104    0.572788  0.555633  0.534734  0.558134  
2      0.901296    0.856754  0.848253  0.768805  0.793120  
3      0.901296    0.929073  0.930780  0.840732  0.837389  
4      0.901296    0.846073  0.813970  0.761217  0.780489  
# # # # # # # # # # 
0.309542221136
0.34862591605
0.156462297163
0.148181744404
0.130663453889
# # # # # # # # # # 

in model: LIGHT_RF  k-fold: 2 / 3

Training until validation scores don't improve for 15 rounds.
[10]	training's auc: 0.858316	valid_1's auc: 0.857418
[20]	training's auc: 0.858334	valid_1's auc: 0.857559
[30]	training's auc: 0.858338	valid_1's auc: 0.857528
Early stopping, best iteration is:
[20]	training's auc: 0.858334	valid_1's auc: 0.857559
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.803331       0.848471    1.000000  0.821575   
1       1            0.635432       0.677809    0.999989  0.660587   
2       1            0.705341       0.739241    0.999999  0.706484   
3       1            0.902340       0.917658    1.000000  0.927674   
4       1            0.847410       0.882805    1.000000  0.865279   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  
0      0.899436    0.796619  0.839949  0.776583  0.785087  
1      0.636157    0.643246  0.639240  0.634272  0.660491  
2      0.636157    0.706532  0.717342  0.685502  0.697619  
3      0.899436    0.898413  0.941636  0.864298  0.845933  
4      0.899436    0.842874  0.897243  0.808239  0.807458  
# # # # # # # # # # 
0.618983052753
0.699348953127
0.311970765203
0.296583250724
0.266176766472
# # # # # # # # # # 

in model: LIGHT_RF  k-fold: 3 / 3

Training until validation scores don't improve for 15 rounds.
[10]	training's auc: 0.858223	valid_1's auc: 0.857434
[20]	training's auc: 0.858241	valid_1's auc: 0.857527
[30]	training's auc: 0.858245	valid_1's auc: 0.857438
Early stopping, best iteration is:
[20]	training's auc: 0.858241	valid_1's auc: 0.857527
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.750734       0.801501         1.0  0.783424   
1       1            0.753294       0.802838         1.0  0.785774   
2       1            0.819101       0.847158         1.0  0.836135   
3       1            0.881146       0.878914         1.0  0.880541   
4       1            0.932505       0.920665         1.0  0.941144   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  
0      0.901261    0.759293  0.785516  0.766816  0.763045  
1      0.901261    0.761090  0.785516  0.771626  0.763045  
2      0.901261    0.827271  0.828132  0.815683  0.801803  
3      0.901261    0.883089  0.855149  0.854510  0.826769  
4      0.901261    0.933076  0.900172  0.898331  0.858630  
# # # # # # # # # # 
0.943463106529
1.04324336316
0.468846338821
0.444293352479
0.402457079334
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.314336       0.287016  2.961617e-06  0.310197      0.375294   
1  1            0.351278       0.345212  3.214177e-05  0.348221      0.375294   
2  2            0.076044       0.087496  1.987238e-11  0.074354      0.122582   
3  3            0.065443       0.077228  9.806616e-12  0.066045      0.122582   
4  4            0.043581       0.063635  3.319671e-12  0.034516      0.122582   

   Neural_net      DART      GOSS  LIGHT_RF  
0    0.315814  0.302949  0.331078  0.314488  
1    0.351463  0.361404  0.365618  0.347748  
2    0.078684  0.091944  0.138685  0.156282  
3    0.069430  0.072333  0.124556  0.148098  
4    0.035471  0.038250  0.101006  0.134152  
AUC train 0.857575382328
AUC train 0.85755946918
AUC train 0.857527223378

in model: LIGHTgbm  k-fold: 1 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858036	valid_1's auc: 0.857535
[20]	training's auc: 0.85815	valid_1's auc: 0.85722
Early stopping, best iteration is:
[1]	training's auc: 0.857864	valid_1's auc: 0.857764
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.479145       0.471692    0.005604  0.440399   
1       1            0.584801       0.511808    0.965718  0.531931   
2       1            0.855217       0.856461    1.000000  0.838500   
3       1            0.931112       0.913982    1.000000  0.916791   
4       0            0.844962       0.844443    1.000000  0.818802   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0      0.375160    0.479597  0.439121  0.461384  0.445780  0.485262  
1      0.642104    0.572788  0.555633  0.534734  0.558134  0.516886  
2      0.901296    0.856754  0.848253  0.768805  0.793120  0.603875  
3      0.901296    0.929073  0.930780  0.840732  0.837389  0.632031  
4      0.901296    0.846073  0.813970  0.761217  0.780489  0.603875  
# # # # # # # # # # 
0.437407503477
0.44792651143
0.368233109324
0.361426377336
0.35111408099
# # # # # # # # # # 

in model: LIGHTgbm  k-fold: 2 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858139	valid_1's auc: 0.857073
[20]	training's auc: 0.858246	valid_1's auc: 0.85657
Early stopping, best iteration is:
[5]	training's auc: 0.858081	valid_1's auc: 0.857632
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.803331       0.848471    1.000000  0.821575   
1       1            0.635432       0.677809    0.999989  0.660587   
2       1            0.705341       0.739241    0.999999  0.706484   
3       1            0.902340       0.917658    1.000000  0.927674   
4       1            0.847410       0.882805    1.000000  0.865279   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0      0.899436    0.796619  0.839949  0.776583  0.785087  0.778484  
1      0.636157    0.643246  0.639240  0.634272  0.660491  0.639334  
2      0.636157    0.706532  0.717342  0.685502  0.697619  0.669229  
3      0.899436    0.898413  0.941636  0.864298  0.845933  0.851720  
4      0.899436    0.842874  0.897243  0.808239  0.807458  0.800182  
# # # # # # # # # # 
0.791229406498
0.823110480744
0.521532946209
0.495500207289
0.4758514671
# # # # # # # # # # 

in model: LIGHTgbm  k-fold: 3 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.858053	valid_1's auc: 0.857517
[20]	training's auc: 0.858173	valid_1's auc: 0.857101
Early stopping, best iteration is:
[5]	training's auc: 0.85799	valid_1's auc: 0.857739
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.750734       0.801501         1.0  0.783424   
1       1            0.753294       0.802838         1.0  0.785774   
2       1            0.819101       0.847158         1.0  0.836135   
3       1            0.881146       0.878914         1.0  0.880541   
4       1            0.932505       0.920665         1.0  0.941144   

   RandomForest  Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0      0.901261    0.759293  0.785516  0.766816  0.763045  0.743136  
1      0.901261    0.761090  0.785516  0.771626  0.763045  0.754224  
2      0.901261    0.827271  0.828132  0.815683  0.801803  0.793811  
3      0.901261    0.883089  0.855149  0.854510  0.826769  0.831197  
4      0.901261    0.933076  0.900172  0.898331  0.858630  0.875505  
# # # # # # # # # # 
1.13150798892
1.18485396146
0.668969371732
0.632595057278
0.597345901146
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.314336       0.287016  2.961617e-06  0.310197      0.375294   
1  1            0.351278       0.345212  3.214177e-05  0.348221      0.375294   
2  2            0.076044       0.087496  1.987238e-11  0.074354      0.122582   
3  3            0.065443       0.077228  9.806616e-12  0.066045      0.122582   
4  4            0.043581       0.063635  3.319671e-12  0.034516      0.122582   

   Neural_net      DART      GOSS  LIGHT_RF  LIGHTgbm  
0    0.315814  0.302949  0.331078  0.314488  0.377169  
1    0.351463  0.361404  0.365618  0.347748  0.394951  
2    0.078684  0.091944  0.138685  0.156282  0.222990  
3    0.069430  0.072333  0.124556  0.148098  0.210865  
4    0.035471  0.038250  0.101006  0.134152  0.199115  
AUC train 0.857763687025
AUC train 0.8576315457
AUC train 0.857738882977

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of rows: 2556790
number of columns: 11

'id',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459140
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of columns: 11
number of data: 2459139
<<<<<<<<<<<<<<<<<<<<
saving DONE.
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
 SAVE  SAVE  SAVE  SAVE  SAVE 
saving df:
dtypes of df:
>>>>>>>>>>>>>>>>>>>>
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of columns: 11
number of data: 2556790
<<<<<<<<<<<<<<<<<<<<
saving DONE.
ON LEVEL: 6

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
DART                  float64
GOSS                  float64
LIGHT_RF              float64
LIGHTgbm              float64
dtype: object
number of rows: 2459140
number of columns: 11

'target',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>
dtypes of df:
id                    category
LogisticRegression     float64
SGDClassifier          float64
GaussianNB             float64
CV                     float64
RandomForest           float64
Neural_net             float64
DART                   float64
GOSS                   float64
LIGHT_RF               float64
LIGHTgbm               float64
dtype: object
number of rows: 2556790
number of columns: 11

'id',
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',

<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<
<<<<<<<<<<<<<<<<<<<<

in model: LogisticRegression  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.446400
1       1            0.539593
2       1            0.809362
3       1            0.894212
4       0            0.782579
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.310522157307
0.336670952843
0.0719194157302
0.0660697838842
0.0475052739872
# # # # # # # # # # 

in model: LogisticRegression  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.807818
1       1            0.604691
2       1            0.678171
3       1            0.930578
4       1            0.867451
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.63386804765
0.677917499756
0.135623808807
0.125118499019
0.0868271153718
# # # # # # # # # # 

in model: LogisticRegression  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression
0       1            0.807760
1       1            0.814880
2       1            0.870737
3       1            0.913738
4       1            0.949858
target                  uint8
LogisticRegression    float64
dtype: object
# # # # # # # # # # 
0.95600586613
1.02424486246
0.205788142781
0.189939681237
0.132350653778
# # # # # # # # # # 
  id  LogisticRegression
0  0            0.318669
1  1            0.341415
2  2            0.068596
3  3            0.063313
4  4            0.044117
AUC train 0.85783870161
AUC train 0.855680970507
AUC train 0.85744327518

in model: SGDClassifier  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  "and default tol will be 1e-3." % type(self), FutureWarning)
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.446400       0.471882
1       1            0.539593       0.527823
2       1            0.809362       0.828769
3       1            0.894212       0.887998
4       0            0.782579       0.813095
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.287583362494
0.331799071229
0.0864831254494
0.0798867227538
0.0659314030879
# # # # # # # # # # 

in model: SGDClassifier  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.807818       0.852681
1       1            0.604691       0.656590
2       1            0.678171       0.732329
3       1            0.930578       0.923032
4       1            0.867451       0.886664
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.585515525658
0.677609624336
0.17061930963
0.157651777564
0.129286145852
# # # # # # # # # # 

in model: SGDClassifier  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier
0       1            0.807760       0.819947
1       1            0.814880       0.824166
2       1            0.870737       0.869189
3       1            0.913738       0.901054
4       1            0.949858       0.929419
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
dtype: object
# # # # # # # # # # 
0.874953495404
1.01238489435
0.252985953684
0.233921306149
0.191704806514
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier
0  0            0.318669       0.291651
1  1            0.341415       0.337462
2  2            0.068596       0.084329
3  3            0.063313       0.077974
4  4            0.044117       0.063902
AUC train 0.857940092787
AUC train 0.857189872303
AUC train 0.857665326297

in model: GaussianNB  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.446400       0.471882    0.009709
1       1            0.539593       0.527823    0.993114
2       1            0.809362       0.828769    1.000000
3       1            0.894212       0.887998    1.000000
4       0            0.782579       0.813095    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
3.68690827489e-06
2.94724394065e-05
2.55070750867e-11
1.44090118385e-11
4.74991806911e-12
# # # # # # # # # # 

in model: GaussianNB  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.807818       0.852681    1.000000
1       1            0.604691       0.656590    0.999984
2       1            0.678171       0.732329    0.999999
3       1            0.930578       0.923032    1.000000
4       1            0.867451       0.886664    1.000000
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
8.45697120475e-06
6.60389429007e-05
6.35809302519e-11
3.59561785151e-11
1.1948753187e-11
# # # # # # # # # # 

in model: GaussianNB  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB
0       1            0.807760       0.819947         1.0
1       1            0.814880       0.824166         1.0
2       1            0.870737       0.869189         1.0
3       1            0.913738       0.901054         1.0
4       1            0.949858       0.929419         1.0
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
dtype: object
# # # # # # # # # # 
1.16033027531e-05
9.11077941211e-05
7.93059599586e-11
4.45934244716e-11
1.46319417319e-11
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB
0  0            0.318669       0.291651  3.867768e-06
1  1            0.341415       0.337462  3.036926e-05
2  2            0.068596       0.084329  2.643532e-11
3  3            0.063313       0.077974  1.486447e-11
4  4            0.044117       0.063902  4.877314e-12
AUC train 0.858025174658
AUC train 0.857752958927
AUC train 0.857965085743

in model: CV  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.446400       0.471882    0.009709  0.454029
1       1            0.539593       0.527823    0.993114  0.561198
2       1            0.809362       0.828769    1.000000  0.822504
3       1            0.894212       0.887998    1.000000  0.902965
4       0            0.782579       0.813095    1.000000  0.815122
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.312003826541
0.353016092802
0.0815481435182
0.0684360489313
0.0412099976101
# # # # # # # # # # 

in model: CV  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.807818       0.852681    1.000000  0.833481
1       1            0.604691       0.656590    0.999984  0.657873
2       1            0.678171       0.732329    0.999999  0.714558
3       1            0.930578       0.923032    1.000000  0.948682
4       1            0.867451       0.886664    1.000000  0.873537
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.62718356307
0.707637637243
0.158854762341
0.132334417725
0.0669828229157
# # # # # # # # # # 

in model: CV  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV
0       1            0.807760       0.819947         1.0  0.782529
1       1            0.814880       0.824166         1.0  0.787782
2       1            0.870737       0.869189         1.0  0.839287
3       1            0.913738       0.901054         1.0  0.898254
4       1            0.949858       0.929419         1.0  0.966462
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
dtype: object
# # # # # # # # # # 
0.938099614927
1.05146171824
0.233450139367
0.192693105491
0.0936222520327
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV
0  0            0.318669       0.291651  3.867768e-06  0.312700
1  1            0.341415       0.337462  3.036926e-05  0.350487
2  2            0.068596       0.084329  2.643532e-11  0.077817
3  3            0.063313       0.077974  1.486447e-11  0.064231
4  4            0.044117       0.063902  4.877314e-12  0.031207
AUC train 0.858008564848
AUC train 0.857730051709
AUC train 0.857929437496

in model: RandomForest  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.446400       0.471882    0.009709  0.454029   
1       1            0.539593       0.527823    0.993114  0.561198   
2       1            0.809362       0.828769    1.000000  0.822504   
3       1            0.894212       0.887998    1.000000  0.902965   
4       0            0.782579       0.813095    1.000000  0.815122   

   RandomForest  
0      0.375041  
1      0.638286  
2      0.898097  
3      0.898097  
4      0.898097  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
0.375040686883
0.375040686883
0.122249648784
0.122249648784
0.122249648784
# # # # # # # # # # 

in model: RandomForest  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.807818       0.852681    1.000000  0.833481   
1       1            0.604691       0.656590    0.999984  0.657873   
2       1            0.678171       0.732329    0.999999  0.714558   
3       1            0.930578       0.923032    1.000000  0.948682   
4       1            0.867451       0.886664    1.000000  0.873537   

   RandomForest  
0      0.899626  
1      0.637917  
2      0.637917  
3      0.899626  
4      0.899626  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
0.753947202018
0.753947202018
0.247513596362
0.247513596362
0.247513596362
# # # # # # # # # # 

in model: RandomForest  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.807760       0.819947         1.0  0.782529   
1       1            0.814880       0.824166         1.0  0.787782   
2       1            0.870737       0.869189         1.0  0.839287   
3       1            0.913738       0.901054         1.0  0.898254   
4       1            0.949858       0.929419         1.0  0.966462   

   RandomForest  
0      0.772246  
1      0.772246  
2      0.901394  
3      0.901394  
4      0.901394  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
dtype: object
# # # # # # # # # # 
1.12928205758
1.12928205758
0.369709762293
0.369709762293
0.369709762293
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest
0  0            0.318669       0.291651  3.867768e-06  0.312700      0.376427
1  1            0.341415       0.337462  3.036926e-05  0.350487      0.376427
2  2            0.068596       0.084329  2.643532e-11  0.077817      0.123237
3  3            0.063313       0.077974  1.486447e-11  0.064231      0.123237
4  4            0.044117       0.063902  4.877314e-12  0.031207      0.123237
AUC train 0.843150167481
AUC train 0.839292947293
AUC train 0.842685639905

in model: Neural_net  k-fold: 1 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.446400       0.471882    0.009709  0.454029   
1       1            0.539593       0.527823    0.993114  0.561198   
2       1            0.809362       0.828769    1.000000  0.822504   
3       1            0.894212       0.887998    1.000000  0.902965   
4       0            0.782579       0.813095    1.000000  0.815122   

   RandomForest  Neural_net  
0      0.375041    0.448610  
1      0.638286    0.540180  
2      0.898097    0.814869  
3      0.898097    0.895169  
4      0.898097    0.791160  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.311647936442
0.338794031081
0.0751881089869
0.0694345656762
0.0333420417915
# # # # # # # # # # 

in model: Neural_net  k-fold: 2 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.807818       0.852681    1.000000  0.833481   
1       1            0.604691       0.656590    0.999984  0.657873   
2       1            0.678171       0.732329    0.999999  0.714558   
3       1            0.930578       0.923032    1.000000  0.948682   
4       1            0.867451       0.886664    1.000000  0.873537   

   RandomForest  Neural_net  
0      0.899626    0.804444  
1      0.637917    0.600357  
2      0.637917    0.678877  
3      0.899626    0.925986  
4      0.899626    0.861578  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.631960274808
0.67992333512
0.142337187454
0.130981159106
0.0625253060677
# # # # # # # # # # 

in model: Neural_net  k-fold: 3 / 3

- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.807760       0.819947         1.0  0.782529   
1       1            0.814880       0.824166         1.0  0.787782   
2       1            0.870737       0.869189         1.0  0.839287   
3       1            0.913738       0.901054         1.0  0.898254   
4       1            0.949858       0.929419         1.0  0.966462   

   RandomForest  Neural_net  
0      0.772246    0.812626  
1      0.772246    0.819718  
2      0.901394    0.872530  
3      0.901394    0.913389  
4      0.901394    0.948321  
target                  uint8
LogisticRegression    float64
SGDClassifier         float64
GaussianNB            float64
CV                    float64
RandomForest          float64
Neural_net            float64
dtype: object
# # # # # # # # # # 
0.952384825275
1.02759860981
0.2154705183
0.198177455442
0.0947201324025
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.318669       0.291651  3.867768e-06  0.312700      0.376427   
1  1            0.341415       0.337462  3.036926e-05  0.350487      0.376427   
2  2            0.068596       0.084329  2.643532e-11  0.077817      0.123237   
3  3            0.063313       0.077974  1.486447e-11  0.064231      0.123237   
4  4            0.044117       0.063902  4.877314e-12  0.031207      0.123237   

   Neural_net  
0    0.317462  
1    0.342533  
2    0.071824  
3    0.066059  
4    0.031573  
AUC train 0.857881642407
AUC train 0.856062084948
AUC train 0.857504682436

in model: DART  k-fold: 1 / 3

/usr/local/lib/python3.5/dist-packages/lightgbm/basic.py:642: UserWarning: max_bin keyword has been found in `params` and will be ignored. Please use max_bin argument of the Dataset constructor to pass this parameter.
  'Please use {0} argument of the Dataset constructor to pass this parameter.'.format(key))
Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857626	valid_1's auc: 0.856409
[20]	training's auc: 0.85767	valid_1's auc: 0.85225
Early stopping, best iteration is:
[8]	training's auc: 0.857597	valid_1's auc: 0.857367
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.446400       0.471882    0.009709  0.454029   
1       1            0.539593       0.527823    0.993114  0.561198   
2       1            0.809362       0.828769    1.000000  0.822504   
3       1            0.894212       0.887998    1.000000  0.902965   
4       0            0.782579       0.813095    1.000000  0.815122   

   RandomForest  Neural_net      DART  
0      0.375041    0.448610  0.425065  
1      0.638286    0.540180  0.550916  
2      0.898097    0.814869  0.842444  
3      0.898097    0.895169  0.898289  
4      0.898097    0.791160  0.846330  
# # # # # # # # # # 
0.297136321683
0.358327992449
0.103351720639
0.0774992526781
0.0691941034041
# # # # # # # # # # 

in model: DART  k-fold: 2 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857793	valid_1's auc: 0.857252
[20]	training's auc: 0.857855	valid_1's auc: 0.857283
Early stopping, best iteration is:
[6]	training's auc: 0.857736	valid_1's auc: 0.8574
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.807818       0.852681    1.000000  0.833481   
1       1            0.604691       0.656590    0.999984  0.657873   
2       1            0.678171       0.732329    0.999999  0.714558   
3       1            0.930578       0.923032    1.000000  0.948682   
4       1            0.867451       0.886664    1.000000  0.873537   

   RandomForest  Neural_net      DART  
0      0.899626    0.804444  0.823902  
1      0.637917    0.600357  0.656353  
2      0.637917    0.678877  0.724123  
3      0.899626    0.925986  0.914643  
4      0.899626    0.861578  0.866939  
# # # # # # # # # # 
0.606253036898
0.74427633157
0.222349540921
0.154582844659
0.112067695317
# # # # # # # # # # 

in model: DART  k-fold: 3 / 3

Training until validation scores don't improve for 20 rounds.
[10]	training's auc: 0.857702	valid_1's auc: 0.85697
[20]	training's auc: 0.857736	valid_1's auc: 0.856656
Early stopping, best iteration is:
[9]	training's auc: 0.857695	valid_1's auc: 0.857653
- - - - - - - - - - 
'LogisticRegression',
'SGDClassifier',
'GaussianNB',
'CV',
'RandomForest',
'Neural_net',
'DART',
'GOSS',
'LIGHT_RF',
'LIGHTgbm',
- - - - - - - - - - 
   target  LogisticRegression  SGDClassifier  GaussianNB        CV  \
0       1            0.807760       0.819947         1.0  0.782529   
1       1            0.814880       0.824166         1.0  0.787782   
2       1            0.870737       0.869189         1.0  0.839287   
3       1            0.913738       0.901054         1.0  0.898254   
4       1            0.949858       0.929419         1.0  0.966462   

   RandomForest  Neural_net      DART  
0      0.772246    0.812626  0.769195  
1      0.772246    0.819718  0.795637  
2      0.901394    0.872530  0.845154  
3      0.901394    0.913389  0.858350  
4      0.901394    0.948321  0.947651  
# # # # # # # # # # 
0.905264798049
1.09985932598
0.32090634111
0.220376997396
0.15536419955
# # # # # # # # # # 
  id  LogisticRegression  SGDClassifier    GaussianNB        CV  RandomForest  \
0  0            0.318669       0.291651  3.867768e-06  0.312700      0.376427   
1  1            0.341415       0.337462  3.036926e-05  0.350487      0.376427   
2  2            0.068596       0.084329  2.643532e-11  0.077817      0.123237   
3  3            0.063313       0.077974  1.486447e-11  0.064231      0.123237   
4  4            0.044117       0.063902  4.877314e-12  0.031207      0.123237   

   Neural_net      DART  
0    0.317462  0.301755  
1    0.342533  0.366620  
2    0.071824  0.106969  
3    0.066059  0.073459  
4    0.031573  0.051788  
AUC train 0.857370052708
AUC train 0.857402089587
AUC train 0.857652810178

in model: GOSS  k-fold: 1 / 3

[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.
Training until validation scores don't improve for 20 rounds.
'''