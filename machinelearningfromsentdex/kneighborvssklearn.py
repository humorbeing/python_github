#http://archive.ics.uci.edu/ml/datasets.html
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
#from sklearn.model_selection import
import pandas as pd

df = pd.read_csv('data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])
accuracies = []
for j in range(25):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test,y_test)
    accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies))
#this is computing very fast.but accuracy is close.
