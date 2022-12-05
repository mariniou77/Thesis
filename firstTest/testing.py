# https://towardsdatascience.com/arima-model-in-python-7bfc7fb792f9

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import DateOffset

df = pd.read_csv("Data/prices.csv", low_memory=False)
print(df.head(30))

df["price_date"] = pd.to_datetime(df["price_date"], dayfirst=True)

# vrisko oles tis egkrafes gia ton nomo atikis
df_Unleaded95 = df.loc[df['county_code'] == 1]

# kratao mono tis stiles pou thelo gia to modelo
df_Unleaded95 = df_Unleaded95[["price_date",
                               "unleaded_95"]]

# theto os index tin imerominia
df_Unleaded95.set_index('price_date', inplace=True)

df = df.sort_values(by=['price_date'])

print(df_Unleaded95.head(30))

# to geniko plot ton timon gia ton nomo atikis ana ta xronia
# basi tou plot tsekaro an einai stationary ta data mou
df_Unleaded95.plot()
# plt.show()

# gia na eimai sigouros gia to stationary tsekaro to p-value < 0.05 (null hypothesis)
result = adfuller(df_Unleaded95['unleaded_95'])
print(dict(zip(['adf',
                'pvalue',
                'usedlag',
                'nobs',
                'critical' 'values',
                'icbest'],
               result)))

# Transform Non-Stationary to Stationary using Differencing (difference(T) = observation(T) — observation(T-1))
df_Unleaded95['1difference'] = df_Unleaded95['unleaded_95'] - \
    df_Unleaded95['unleaded_95'].shift(1)
df_Unleaded95['1difference'].plot()
plt.show()

# tsekaro xana to p-value an einai < 0.05. an den einai sinexizo to tranformation mexri na einai
result = adfuller(df_Unleaded95['1difference'].dropna())
print(dict(zip(['adf',
                'pvalue',
                'usedlag',
                'nobs',
                'critical' 'values',
                'icbest'],
               result)))


# df_Unleaded95['Seasonal_Difference'] = df_Unleaded95['unleaded_95'] - \
#     df_Unleaded95['unleaded_95'].shift(1741)
# df_Unleaded95['Seasonal_Difference'].plot()
# plt.show()

# result = adfuller(df_Unleaded95['Seasonal_Difference'].dropna())
# print(dict(zip(['adf',
#                 'pvalue',
#                 'usedlag',
#                 'nobs',
#                 'critical' 'values',
#                 'icbest'],
#                result)))

# df_Unleaded95['1Seasonal_Difference'] = df_Unleaded95['Seasonal_Difference'] - \
#     df_Unleaded95['Seasonal_Difference'].shift(1)
# df_Unleaded95['1Seasonal_Difference'].plot()
# plt.show()

# # tsekaro xana to p-value an einai < 0.05. an den einai sinexizo to tranformation mexri na einai
# result = adfuller(df_Unleaded95['1difference'].dropna())
# print(dict(zip(['adf',
#                 'pvalue',
#                 'usedlag',
#                 'nobs',
#                 'critical' 'values',
#                 'icbest'],
#                result)))

fig1 = plot_acf(df_Unleaded95['1difference'].dropna())
plt.show()
fig2 = plot_pacf(df_Unleaded95['1difference'].dropna())
# plt.show()


# fig1 = plot_acf(df_Unleaded95['Seasonal_Difference'].dropna())
# plt.show()
# fig2 = plot_pacf(df_Unleaded95['Seasonal_Difference'].dropna())
# plt.show()

model = SARIMAX(df["unleaded_95"].dropna(), order=(2,1,1), trend='c')
result = model.fit()
result.resid.plot(kind='kde')
# plt.show()


new_dates = [df_Unleaded95.index[-1]+DateOffset(days=x) for x in range(1, 180)]
df_Unleaded95_pred = pd.DataFrame(
    index=new_dates, columns=df_Unleaded95.columns)
print(df_Unleaded95_pred.head())

df2 = pd.concat([df_Unleaded95, df_Unleaded95_pred])
# we have 198 rows that's why we start at 199
df2['predictions'] = result.get_forecast(start=1600,end=1800).predict_mean
df2[['unleaded_95', 'predictions']].plot()
plt.show()


# print(df_Unleaded95.tail(30))

# print(df2)