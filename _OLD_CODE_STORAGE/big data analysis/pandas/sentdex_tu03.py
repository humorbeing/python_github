import pandas as pd

df = pd.read_csv('ZILLOW-Z77006_ZRISFRR.csv')
print(df.head())
# df2 = pd.read_csv('members.csv')
# print(df2.head())
# print(df2.tail())
df.set_index('Date', inplace=True)
df.to_csv('date_index.csv')
df = pd.read_csv('date_index.csv')
print(df.head())
df = df = pd.read_csv('date_index.csv', index_col=0)
print(df.head())

df.columns = ['HPI']
print(df.head())

df.to_csv('new2.csv')
df.to_csv('new3.csv', header=False)
df = pd.read_csv('new2.csv', index_col=0)
print(df.head())
df = pd.read_csv('new3.csv')
print(df.head())
df = pd.read_csv('new3.csv', names=['left', 'right'], index_col=0)
print(df.head())

df.to_html('web.html')

df.rename(columns={'right':'haha'}, inplace=True)
print(df.head())