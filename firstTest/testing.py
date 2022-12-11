from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.stattools import acf

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

kafsimo = 'unleaded_95'
nomos = 'ΝΟΜΟΣ_ΑΡΤΗΣ'
periodos = 180

print(kafsimo)
print(nomos)
print(periodos)
# Import data
df = pd.read_csv("Data/prices.csv", low_memory=False)

df['validation'] = df['validation'].str.replace(' ', '_')

df = df.loc[df['validation'] == nomos]

df = df[["record_date",
        kafsimo]]

df = df.reset_index()
df = df.drop(columns=['index'])
df.record_date = pd.to_datetime(df.record_date)

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df[kafsimo]); axes[0, 0].set_title('Original Series')
plot_acf(df[kafsimo], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df[kafsimo].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df[kafsimo].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df[kafsimo].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df[kafsimo].diff().diff().dropna(), ax=axes[2, 1])

plt.show()
result = adfuller(df[kafsimo].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

y = df[kafsimo]

## Adf Test
ndiffs(y, test='adf')  # 2

# KPSS test
ndiffs(y, test='kpss')  # 0

# PP test:
ndiffs(y, test='pp')  # 2

# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df[kafsimo].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df[kafsimo].diff().dropna(), ax=axes[1])

plt.show()

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df[kafsimo].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df[kafsimo].diff().dropna(), ax=axes[1])

plt.show()

# 1,1,2 ARIMA Model
model = SARIMAX(df[kafsimo], order=(2,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

plot_predict(model_fit, 1280, 1400)
plt.show()

train = df[kafsimo][:1280-periodos]
test = df[kafsimo][1280-periodos:]

model = SARIMAX(train, order=(1, 1, 1))  
fitted = model.fit(disp=-1)  

# Forecast
fc = fitted.forecast(periodos, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

################################

model = SARIMAX(df[kafsimo], order=(1, 1, 1))  
fitted = model.fit(disp=-1)  

# Forecast
fc = fitted.forecast(periodos, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(df[kafsimo], label='training')
# plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()