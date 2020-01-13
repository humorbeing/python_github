import pandas as pd
import quandl #why in video it is Quandl? trid pip install requests
import math

df = quandl.get('WIKI/GOOGL') #dataframe
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True) #out liar,  you don't want to get rid of data if one missing.

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

print(df.head())
#df.dropna(inplace=True)
#print(df.tail())
