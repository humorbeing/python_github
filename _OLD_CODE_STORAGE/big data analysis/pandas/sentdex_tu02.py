import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

web_stats = {'Day':[1,2,3,4,5,6],
             'Visitors':[43,34,65,56,29,76],
             'Bounce_Rate':[65,67,78,65,45,52]}

df = pd.DataFrame(web_stats)

# print(df)
# print(df.head())
# print(df.tail())
print(df.tail(2))
cf = df.set_index('Day')
print(df)  # doesn't work
# print(cf.head())
# print(cf.tail())
print(cf.tail(2))

df.set_index('Day', inplace=True)

print(df.tail(2))
# print(df['Day'])  # can't call index??
print(df['Visitors'])
print(df.Visitors)
# print(df.Day)
print(df[['Bounce_Rate', 'Visitors']])

print(df.Visitors.tolist())
print(np.array(df[['Bounce_Rate', 'Visitors']]))

df2 = pd.DataFrame(np.array(df[['Bounce_Rate', 'Visitors']]))
print(df2)