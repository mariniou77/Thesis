# https://www.geeksforgeeks.org/python-arima-model-for-time-series-forecasting/

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("fuel_prices.csv", low_memory=False)

df = df.drop(columns=["id", "heatingoil"])

df["date"] = pd.to_datetime(df["date"], dayfirst=True)

df = df.sort_values(by=['date', "nomos"])

df = df[["nomos",
         "unleaded95",
         "date",
         "validation",
         "geography",
         "perifereia"]]

df_Unleaded95 = df.loc[df['nomos'] == 1]
df = df_Unleaded95[["date",
                    "unleaded95"]]
df.set_index('date', inplace=True)

print(df.head(30))

plt.plot(df[["date"]], df["unleaded95"])
plt.xticks(rotation=90)

