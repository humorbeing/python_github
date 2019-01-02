from sklearn import svm, datasets
from sklearn import metrics
import pandas as pd
import numpy as np

clf = svm.SVC()

X = [
    [0, 0, 0],
    [0, 1, 1],
    [0, 0, 1],
    # [1, 0],
    [1, 1, 1],
]
Y = [
    0,
    1,
    0,
    1,
]
X_t = pd.DataFrame(X)
Y_t = pd.DataFrame(Y)

print(X_t)
print(Y_t)
clf.fit(X_t, Y)
print(clf.predict(X))