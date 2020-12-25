# %% Import Libraries
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt

plt.style.use('./plot.mplstyle')
# %% read data

df = pd.read_csv("./data/portfolio.csv")

df.replace(to_replace="-", value="0", inplace=True)
df.replace(to_replace="%", value="", inplace=True, regex=True)

df1 = df.iloc[:, 1:].astype("float")

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_array = scaler.fit_transform(df1)


final_df = pd.DataFrame(scaled_array, index=df1.index, columns=df1.columns)

final_df.insert(0, 'Symbol', df['Symbol'])


# %% print heatmap using seaborn

sns.heatmap(final_df.corr())
