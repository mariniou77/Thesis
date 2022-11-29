from statsmodels.tsa.stattools import adfuller
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_predict 

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# kafsimo = str(sys.argv[1])
# nomos = str(sys.argv[2])
# periodos = int(sys.argv[3])

kafsimo = 'unleaded_95'
nomos = 'ΝΟΜΟΣ_ΑΡΤΗΣ'
periodos = 6

print(kafsimo)
print(nomos)
print(periodos)

df = pd.read_csv("Data/prices.csv", low_memory=False)

df['validation'] = df['validation'].str.replace(' ', '_')

# vrisko oles tis egkrafes gia ton nomo pou dothike
df = df.loc[df['validation'] == nomos]

# kratao mono tis stiles pou thelo gia to modelo
df = df[["record_date",
        kafsimo]]

# theto os index tin imerominia
df.set_index('record_date', inplace=True)

df = df.sort_values(by=['record_date'])

print(df.head(15))

result = adfuller(df[kafsimo].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

result = adfuller(df[kafsimo].diff().diff().dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


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

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df[kafsimo].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df[kafsimo].diff().dropna(), ax=axes[1])

plt.show()

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df[kafsimo].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df[kafsimo].diff().dropna(), ax=axes[1])

plt.show()

# 1,1,2 ARIMA Model
model = sm.tsa.arima.ARIMA(df, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

fig, ax = plt.subplots()
ax = df.loc['2018-01-01':].plot(ax=ax)
plot_predict(model_fit, '2018-01-01', '2022-02-07', ax=ax)
plt.show()