import pandas as pd
import numpy as np
import time


since = time.time()

df = pd.DataFrame(np.arange(0,10).reshape(5,2),
                  index=range(0, 10, 2),
                  columns=list('AB'))

print(df)
df1 = df.iloc[[2]]
df2 = df.loc[[2]]
df3 = df.loc[[2,4,6]]
b = set([2,4,6])
b1 = set([2,3,4,6])
df4 = df.loc[b]
print(b)
print(df1)
print(df2)
print(df3)
print('df4', df4)
print('df41',df.loc[b1])
print('df5',df[df.A == 3])
print('df6',df[df.A == 2])
print(df.A)
print(df.A==2)
print(df.B==1)
print('ttt', df.A ==b)
print('aaa')
# print(df.set_index('A').loc[b])
# dfa = df.set_index('A')
dfa = df.loc[df['A'].isin(b)]
print(dfa)
dfb = dfa.set_index('B')
dfb = dfa.loc[dfa['B'].isin(['3'])]
print(dfb)
# print(df.set_index('B'))
# print('---',df.reindex(b1))
# df = df.reindex(b1)
# print(df.dropna())
# print('bbb')
# print(df[df.A == b])
print()
time_elapsed = time.time() - since
print('[timer]: complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))