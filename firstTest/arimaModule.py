# https://github.com/ggagnon1995/ARIMA_Algorithm/blob/main/ARIMA%20Algorithm.ipynb 

import pandas as pd
import sys
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.seasonal import seasonal_decompose
 
# kafsimo = str(sys.argv[1])
# nomos = str(sys.argv[2])
# periodos = int(sys.argv[3])

kafsimo = 'unleaded_95'
nomos = 'ΝΟΜΟΣ_ΑΡΤΗΣ'
periodos = 180

print(kafsimo)
print(nomos)
print(periodos)

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

def ARIMA_parameter_optimization(df):
    
    d = differencing_parameter(df)
    
    order_aic_bic =[]
    # Loop over AR order
    for p in range(4):
        # Loop over MA order
        for q in range(4):
            # Fit models
            model = SARIMAX(df[kafsimo].dropna(), order=(p,d,q), trend='c', seasonal_order=(p, d, q, 12))
            results = model.fit()
            # Add order and statistics to list
            order_aic_bic.append((p, d, q, results.aic, results.bic))
    
    # Save parameters to a data frame 
    order_df = pd.DataFrame(order_aic_bic, columns=['p','d','q', 'aic', 'bic'])
    
    # Select the parameters with the lowest AIC 
    parameter_df = order_df.loc[(order_df['aic'] == order_df['aic'].min())]
    
    return parameter_df    


def ARIMA_model(df):
    
    parameter_df = ARIMA_parameter_optimization(df)
    
    p = parameter_df['p'].mean()
    q = parameter_df['q'].mean()
    d = parameter_df['d'].mean()

    model = SARIMAX(df[kafsimo].dropna(), order=(p,d,q), trend = 'c', seasonal_order=(p, d, q, 12))
    model_fit = model.fit()
    
    return model_fit

def out_of_sample_fcast(df):
    
    model_fit = ARIMA_model(df)
    output = model_fit.forecast()
    future_forecast = model_fit.get_forecast(steps=periodos).predicted_mean
    df_forecast = future_forecast.to_frame()
    
    return df_forecast

def final_frame(df):
    
    df_forecast = out_of_sample_fcast(df)
    date_range = pd.DatetimeIndex(df['record_date']) + pd.Timedelta(days=periodos)

    date_range = date_range.to_frame() # save to a frame
    date_range['dotw'] = date_range['record_date'].dt.dayofweek # add day of week

    df = df.set_index('record_date') # set the index in the original data frame to the date colum
    df = date_range.join(df) # merge
    df = df.drop(['record_date'], axis=1).reset_index() # reset the index
    
    final_df = df_forecast.join(df)
    
    return final_df

df = pd.read_csv("Data/prices.csv", low_memory=False)

df['validation'] = df['validation'].str.replace(' ', '_')

# vrisko oles tis egkrafes gia ton nomo pou dothike
df = df.loc[df['validation'] == nomos]

# kratao mono tis stiles pou thelo gia to modelo
df = df[["record_date",
        kafsimo]]

# theto os index tin imerominia
# df.set_index('record_date', inplace=True)
# df["record_date"] = pd.to_datetime(df["record_date"], dayfirst=True)

# df = df.sort_values(by=['record_date'])

# print(df.head(30))

df_testing = df[["record_date",
        kafsimo]][:1280-periodos]

# dh = df.asfreq('D')
# decomposition = sm.tsa.seasonal_decompose(df[kafsimo], model='additive', 
#                             extrapolate_trend='freq') #additive or multiplicative is data specific
# fig = decomposition.plot()
# plt.show()

results_df = final_frame(df)

results_df_testing = final_frame(df_testing)
dh = df

dh['record_date'] = pd.to_datetime(dh['record_date'])
dh = dh.set_index('record_date').asfreq('D')
result = seasonal_decompose(dh, model='ad')

from pylab import rcParams
rcParams['figure.figsize'] = 12,5
result.plot()

print(results_df)

results_df.plot()
plt.show()


# προσπαθω να βρω ποσες μερες λειπουν απο το dataframe και να τις γεμισω 
# με την τιμη του καυσιμου της προηγουμενης μερας
# με σκοπο να δω αν υπαρχει καποιο seasonality 

dm = df
dm.reset_index(inplace = True, drop = True)

start_date = pd.to_datetime("2018-01-01")
end_date = pd.to_datetime("2022-10-26")

all_date = pd.date_range(start_date, end_date, freq='d')

all_date_df = pd.DataFrame({'record_date':all_date})
all_date_df[kafsimo] = ""
for index in range(1279):
    if(all_date_df['record_date'][index] == dm['record_date'][index]):
        all_date_df[kafsimo][index] == dm[kafsimo][index]
    else:
         all_date_df[kafsimo][index] = ""   

# df.set_index('record_date', inplace=True)

# results_df = results_df.drop(['dotw', 'record_date'], axis=1)

# new_dates = [df.index[-1]+DateOffset(days=x) for x in range(1, periodos+1)]

# df_Unleaded95_pred = pd.DataFrame(
#     index=new_dates, columns=df.columns)

# df_Unleaded95_pred["predictions"] = results_df['predicted_mean'].values

# df2 = pd.concat([df, df_Unleaded95_pred])

# df2[['unleaded_95', 'predictions']].plot()

# print(df_Unleaded95_pred.head())