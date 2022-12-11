# https://analyticsindiamag.com/complete-guide-to-sarimax-in-python-for-time-series-modeling/  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
 
# kafsimo = str(sys.argv[1])
# nomos = str(sys.argv[2])
# periodos = int(sys.argv[3])

kafsimo = 'unleaded_95'
nomos = 'ΝΟΜΟΣ_ΑΡΤΗΣ'
periodos = 600

print(kafsimo)
print(nomos)
print(periodos)

df = pd.read_csv("Data/prices.csv", low_memory=False)

df['validation'] = df['validation'].str.replace(' ', '_')

# vrisko oles tis egkrafes gia ton nomo pou dothike
df = df.loc[df['validation'] == nomos]

# kratao mono tis stiles pou thelo gia to modelo
df = df[["record_date", kafsimo]]
df["record_date"] = pd.to_datetime(df["record_date"], dayfirst=True)
df = df.sort_values(by=['record_date'])

# fill the missing dates of the dataframe
dm = df
dm.set_index('record_date', inplace=True)
idx = pd.date_range(dm.index.min(), dm.index.max())
dm.index = pd.DatetimeIndex(dm.index)
dm = dm.reindex(idx)

# df.plot()
# plt.show()

# dm.plot()
# plt.show()

#Interpolate in forward order across the column:
dm.interpolate(method ='linear', limit_direction ='forward', inplace=True)

# searching for seasonality
dh = dm.asfreq('M') #for daily resampled data and fillnas with appropriate method
decomposition = sm.tsa.seasonal_decompose(dh[kafsimo], model='additive', 
                            extrapolate_trend='freq') #additive or multiplicative is data specific
fig = decomposition.plot()
plt.show()

seasonality=decomposition.seasonal
seasonality.plot(color='green')
plt.show()

dftest = adfuller(dm[kafsimo], autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

# checking the d parameter
# first try
rolling_mean = dm.rolling(window = 12).mean()
dm['rolling_mean_diff'] = rolling_mean - rolling_mean.shift()
ax1 = plt.subplot()
dm['rolling_mean_diff'].plot(title='after rolling mean & differencing')
ax2 = plt.subplot()
dm.plot(title='original')

dftest = adfuller(dm['rolling_mean_diff'].dropna(), autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
  print("\t",key, ": ", val)

# trying ARIMA model
model=sm.tsa.arima.ARIMA(dm[kafsimo], order=(1,1,1))
history=model.fit()

history.summary()

dm['forecast']=history.predict(start=1000,end=1279,dynamic=True)
dm[[kafsimo,'forecast']].plot(figsize=(12,8))

# stationar data given
model=sm.tsa.arima.ARIMA(dm['rolling_mean_diff'].dropna(),order=(1,1,1))
model_fit=model.fit()

dm['forecast2']=model_fit.predict(start=1000,end=1279,dynamic=True)
dm[['rolling_mean_diff','forecast2']].plot(figsize=(12,8))

# SARIMAX
model=sm.tsa.statespace.SARIMAX(dm[kafsimo],order=(1, 1, 1),seasonal_order=(6,1,1,12))
results=model.fit()

dm['forecast']=results.predict(start=900,end=1030,dynamic=True)
dm[[kafsimo,'forecast']].plot(figsize=(12,8))

# Making a NAN value future dataset
pred_date=[dm.index[-1]+ DateOffset(days=x) for x in range(0,periodos)]
pred_date=pd.DataFrame(index=pred_date[1:],columns=dm.columns)

data=pd.concat([dm,pred_date])
data['forecast'] = results.predict(start = 1760, end = 2200, dynamic= True)  
data[[kafsimo, 'forecast']].plot(figsize=(12, 8))
plt.show()