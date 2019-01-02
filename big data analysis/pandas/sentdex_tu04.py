import pandas as pd

df = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
# print(df.head())  # its a list
# print(df)

# print(df[0])
print(df[0][0])

for abb in df[0][2][1:]:
    print('welcome to '+abb)