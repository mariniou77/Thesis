# https://towardsdatascience.com/arima-model-in-python-7bfc7fb792f9

import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from pandas.tseries.offsets import DateOffset

df = pd.read_csv("prices.csv", low_memory=False)
print(df.head(30))

# df = df.drop(columns=["id", "heatingoil"])
#
df["price_date"] = pd.to_datetime(df["price_date"], dayfirst=True)
#
df = df.sort_values(by=['price_date'])

# df = df[["nomos",
#           "unleaded95",
#           "date",
#           "validation",
#           "geography",
#           "perifereia"]]
#
df_Unleaded95 = df.loc[df['county_code'] == 1]
df_Unleaded95 = df_Unleaded95[["price_date",
                                "unleaded_95"]]
df_Unleaded95.set_index('price_date', inplace=True)
#
#
#
# print(df_Unleaded95.head(30))
#
# df_Unleaded95.plot()
# plt.show()
# # df_Unleaded95.plot(x='date', y='unleaded95')
# result = adfuller(df_Unleaded95['unleaded95'])
# # to help you, we added the names of every value
# print(dict(zip(['adf',
#                 'pvalue',
#                 'usedlag',
#                 'nobs',
#                 'critical' 'values',
#                 'icbest'],
#                result)))
#
#
# df_Unleaded95['1difference'] = df_Unleaded95['unleaded95'] - \
#     df_Unleaded95['unleaded95'].shift(1)
# df_Unleaded95['1difference'].plot()
# plt.show()
#
# result = adfuller(df_Unleaded95['1difference'].dropna())
# # to help you, we added the names of every value
# print(dict(zip(['adf',
#                 'pvalue',
#                 'usedlag',
#                 'nobs',
#                 'critical' 'values',
#                 'icbest'],
#                result)))
#
# df_Unleaded95['2difference'] = df_Unleaded95['1difference']-df_Unleaded95['1difference'].shift(1)
# df_Unleaded95['2difference'].plot()
# plt.show()
#
# result = adfuller(df_Unleaded95['2difference'].dropna())
# # to help you, we added the names of every value
# print(dict(zip(['adf',
#                 'pvalue',
#                 'usedlag',
#                 'nobs',
#                 'critical' 'values',
#                 'icbest'],
#                result)))
#
# df_Unleaded95['Seasonal_Difference']=df_Unleaded95['unleaded95']-df_Unleaded95['unleaded95'].shift(1)
# df_Unleaded95['Seasonal_Difference'].plot()
# plt.show()
#
# # result = adfuller(df_Unleaded95['Seasonal_Difference'].dropna())
# # # to help you, we added the names of every value
# # print(dict(zip(['adf',
# #                 'pvalue',
# #                 'usedlag',
# #                 'nobs',
# #                 'critical' 'values',
# #                 'icbest'],
# #                result)))
#
# fig1 = plot_acf(df_Unleaded95['2difference'].dropna())
# plt.show()
# fig2 = plot_pacf(df_Unleaded95['2difference'].dropna())
# plt.show()
#
# model=SARIMAX(df_Unleaded95['unleaded95'],order=(1,2,1),seasonal_order=(1, 0, 0, 12))
# result=model.fit()
# result.resid.plot(kind='kde')
# plt.show()
#
#
# new_dates=[df_Unleaded95.index[-1]+DateOffset(days=x) for x in range(1,48)]
# df_Unleaded95_pred=pd.DataFrame(index=new_dates,columns =df_Unleaded95.columns)
# print(df_Unleaded95_pred.head())
#
# df2=pd.concat([df_Unleaded95,df_Unleaded95_pred])
# # we have 198 rows that's why we start at 199
# df2['predictions']=result.predict(start=100,end=245)
# df2[['unleaded95','predictions']].plot()
# plt.show()