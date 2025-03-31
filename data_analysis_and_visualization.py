import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


df = pd.read_csv('data_set/sec_data.csv')
# df.columns = df.columns.str.replace(' ', '_')

# df = df.drop(df.columns[[0,1,2,3,4]], axis=1)

# df['sma'] = df['open'].rolling(window=5).mean()
df['ema'] = df['open'].ewm(span=5).mean()
df['resistance'] = df['high'].apply(lambda x:x+2).ewm(span=5).mean()
df['support'] = df['low'].apply(lambda x:x-5).ewm(span=5).mean()

# print(df.head())
# df.dropna(inplace=True, axis=0)

# df = df.reset_index(drop=True)

# print(df.isna().sum())
# print(df.shape)
# print(df.describe)
# print(df.head())
# print(df.tail())

# visualization
# sns.heatmap(df.corr(), annot=True, cmap='Blues', center=0, linewidths=.2)
sns.scatterplot(df, x='time', y='open', hue='high', size='low')
plt.plot(df['time'], df['open'])
plt.plot(df['time'], df['resistance'])
plt.plot(df['time'], df['support'])


plt.show()