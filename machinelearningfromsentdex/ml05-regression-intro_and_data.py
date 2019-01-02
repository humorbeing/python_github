import pandas as pd
import quandl #why in video it is Quandl? trid pip install requests
import math, datetime
import numpy as np #let us use arrays, python don't really have arrays
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL') #dataframe
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True) #out liar,  you don't want to get rid of data if one missing.

forecast_out = int(math.ceil(0.01*len(df))) #0.1 or 0.01
#print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.head())
#print(df.tail())

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] #maked mistake here.
X = X[:-forecast_out]        #this is right order sent fixed.


 #scale all, not only one.
#X = X[:-forecast_out+1]
#df.dropna(inplace=True)
df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label']) # mistake

#print(len(X),len(y)) #match up X,y

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression() #0.97, n_jobs ,how many run at once. -1 is no limit.
#clf = svm.SVR() # 0.82
#clf = svm.SVR(kernel='poly') #0.71  support vactor machines?
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set,accuracy,forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix =last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#print(accuracy)
