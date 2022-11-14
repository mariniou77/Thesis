# https://www.geeksforgeeks.org/python-arima-model-for-time-series-forecasting/

import pandas as pd
import matplotlib.pyplot as plt

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

print(df_Unleaded95)

df_Unleaded95.plot(x='date', y='unleaded95')

plt.show()
