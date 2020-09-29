import pandas as pd
import numpy as np

from numpy.random import randn

# series
labels = ['name', 'surname', 'age']
data = np.array(['Lorenzo', 'Conti', 23])
series = pd.Series(data=data, index=labels)

print(series)
print(series['name'])

# sum of series
print(pd.Series([10, 10, 2020], ['day', 'month', 'year']) + pd.Series(pd.Series([5, 0], ['day', 'month'])))

# dataframe
np.random.seed(101)

df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])
print(df)

print(df['W'])  # or print(df.W)
print(type(df['W']))
print(df[['W', 'Z']])

df['Sum'] = df['W'] + df['Y']
print(df)

# inplace drop columns
df.drop('Sum', axis=1, inplace=True)

# inplace drop rows
df.drop('A', axis=0, inplace=True)

# row series
print(df.loc['B'])
print(df.iloc[2])

print(df.loc[['B', 'C'], ['W', 'Y']])

# selection
print(df[df < 1])

