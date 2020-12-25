# %% Import Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('./plot.mplstyle')

df = pd.read_csv('./data/NFLX_news.csv')
df['date'] = pd.to_datetime(df['datetime'], unit='ms')

news_df = df.resample('D', on='date').count()[['headline']]

# %% Plot NFLX stock

df_stock = pd.read_csv('./data/NFLX.csv')
df_stock['date'] = pd.to_datetime(df_stock['startEpochTime'], unit='s')

stock_df = df_stock.resample('D', on='date').mean()[['openPrice']]

# %% Merge two dataframes
merged_df = stock_df.merge(news_df, left_on='date',
                           right_on='date', suffixes=(False, False))

merged_df = (merged_df.ffill()+merged_df.bfill())/2

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(merged_df)

pd.DataFrame(scaled_data, columns=["openPrice", "news_count"]).plot()

pd.DataFrame(scaled_data, columns=["openPrice", "news_count"]).corr()
