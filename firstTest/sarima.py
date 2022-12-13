# https://analyticsindiamag.com/complete-guide-to-sarimax-in-python-for-time-series-modeling/  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
 
# kafsimo = str(sys.argv[1])
# nomos = str(sys.argv[2])
# periodos = int(sys.argv[3])

def differencing_parameter(df):
    
    first_diff = adfuller(df[kafsimo].dropna())
    second_diff = adfuller(df[kafsimo].diff().dropna())

    if ((first_diff[1] > 0.05) and (second_diff[1] > 0.05)): 
        d = 2
    elif((first_diff[1] > 0.05) and (second_diff[1] <= 0.05)):
        d =1 
    else: 
        d = 0 
        
    return d 

def differencing_parameter_seasonal(df):
    
    first_diff = adfuller(df[kafsimo].diff(12).dropna())
    second_diff = adfuller(df[kafsimo].diff(12).diff().dropna())

    if ((first_diff[1] > 0.05) and (second_diff[1] > 0.05)): 
        d = 2
    elif((first_diff[1] > 0.05) and (second_diff[1] <= 0.05)):
        d =1 
    else: 
        d = 0 
        
    return d 

kafsimo = 'unleaded_95'
nomos = 'ΝΟΜΟΣ_ΑΡΤΗΣ'
periodos = 180

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

second_diff = adfuller(df[kafsimo].diff(12).dropna())
print(second_diff[1])

# fill the missing dates of the dataframe
dm = df
dm.set_index('record_date', inplace=True)
idx = pd.date_range(dm.index.min(), dm.index.max())
dm.index = pd.DatetimeIndex(dm.index)
dm = dm.reindex(idx)

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

#####################################################################################################
#### PREVIOUS PREDICTION ####

t_dm = dm
split_value = dm.size - periodos
training_dm = dm.iloc[:split_value, :]

e_d = differencing_parameter(training_dm)
e_dD = differencing_parameter_seasonal(training_dm)

existing_model=sm.tsa.statespace.SARIMAX(training_dm[kafsimo],order=(1, e_d, 1),seasonal_order=(6,e_dD,1,12))
existing_results = existing_model.fit()

t_dm['forecast'] = existing_results.predict(start = split_value, end = t_dm.size, dynamic= True)  
t_dm[[kafsimo, 'forecast']].plot(figsize=(12, 8))
plt.show()


#####################################################################################################
#### FUTURE PREDICTION ####
# SARIMAX
f_d = differencing_parameter(dm)
f_dD = differencing_parameter_seasonal(dm)
future_model=sm.tsa.statespace.SARIMAX(dm[kafsimo],order=(1, f_d, 1),seasonal_order=(6,f_dD,1,12))
future_results = future_model.fit()

# Making a NAN value future dataset
pred_date = [dm.index[-1]+ DateOffset(days=x) for x in range(0,periodos)]
pred_date = pd.DataFrame(index=pred_date[1:],columns=dm.columns)

data=pd.concat([dm,pred_date])
p_start = dm.size
p_end = dm.size + periodos
data['forecast'] = future_results.predict(start = p_start, end = p_end, dynamic= True)  
data[[kafsimo, 'forecast']].plot(figsize=(12, 8))
plt.show()