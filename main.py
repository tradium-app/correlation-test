# %% Import Libraries
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt

plt.style.use('./plot.mplstyle')
# %% read data


def strip_percentage(series):
    return series.str.rstrip('%')


df = pd.read_csv("./data/portfolio.csv")
df.replace(to_replace="-", value="0", inplace=True)
df['Revenue YoY'] = strip_percentage(df['Revenue YoY'])
df['Insider %'] = strip_percentage(df['Insider %'])
df['Last Price Vs. 50D SMA'] = strip_percentage(df['Last Price Vs. 50D SMA'])
df['Revenue FWD'] = strip_percentage(df['Revenue FWD'])
df['6M Perf'] = strip_percentage(df['6M Perf'])
df['YTD Perf'] = strip_percentage(df['YTD Perf'])
df['1M Perf'] = strip_percentage(df['1M Perf'])

df1 = df.iloc[:, 1:].astype("float")

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_array = scaler.fit_transform(df1)

type(scaled_array)

final_df = pd.DataFrame(scaled_array, index=df1.index, columns=df1.columns)

final_df.insert(0, 'Symbol', df['Symbol'])


# %% print heatmap using seaborn

sns.heatmap(final_df.corr())
