# https://www.youtube.com/watch?v=pryXhOgDY9A

from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset

# kafsimo = str(sys.argv[1])
# nomos = str(sys.argv[2])
# periodos = int(sys.argv[3])

kafsimo = 'unleaded_95'
nomos = 'ΝΟΜΟΣ_ΑΡΤΗΣ'
periodos = 180

print(kafsimo)
print(nomos)
print(periodos)

# function to find the best 'd' parameter for the model
def differencing_parameter(set_df):
    
    first_diff = adfuller(set_df)
    dsf = pd.DataFrame(set_df, columns=[kafsimo])
    s_set = list(dsf[kafsimo].diff().dropna())
    second_diff = adfuller(s_set)

    if ((first_diff[1] > 0.05) and (second_diff[1] > 0.05)): 
        d = 2
    elif((first_diff[1] > 0.05) and (second_diff[1] <= 0.05)):
        d =1 
    else: 
        d = 0 
        
    return d 

# function to find the best 'p' and 'q' for the model
def ARIMA_parameter_optimization(set_df):
    
    d = differencing_parameter(set_df)
    
    order_aic_bic =[]
    # Loop over AR order
    for p in range(4):
        # Loop over MA order
        for q in range(4):
            # Fit models
            model = ARIMA(set_df, order=(p, d, q)) #, seasonal_order=(p, d, q, 12))
            results = model.fit()
            # Add order and statistics to list
            order_aic_bic.append((p, d, q, results.aic, results.bic))
    
    # Save parameters to a data frame 
    order_df = pd.DataFrame(order_aic_bic, columns=['p','d','q', 'aic', 'bic'])
    
    # Select the parameters with the lowest AIC 
    parameter_df = order_df.loc[(order_df['aic'] == order_df['aic'].min())]
    
    return parameter_df     

#  function to find the predictions of the existing values based on the time of period 
#  the user gave
def previous_prediction(df, periodos):

    training_set, testing_set = split_dataframe(df, periodos)
    model_existing_predictions = []

    parameter_df = ARIMA_parameter_optimization(training_set)
    
    p = parameter_df['p'].mean()
    q = parameter_df['q'].mean()
    d = parameter_df['d'].mean()

    print("pp = " + str(p))
    print("pq = " + str(q))
    print("pd = " + str(d))

    for i in range(periodos): 
        model = ARIMA(training_set, order=(p, d, q)) #, seasonal_order=(p, d, q, 12))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_existing_predictions.append(yhat)
        actual_test_value = testing_set[i]
        training_set.append(actual_test_value)

    return model_existing_predictions, training_set, model    

# function to predict the future values
def future_prediction(df, periodos):

    model_future_predictions = []
    set_df = list(df[kafsimo])
    
    parameter_df = ARIMA_parameter_optimization(set_df)
    
    p = parameter_df['p'].mean()
    q = parameter_df['q'].mean()
    d = parameter_df['d'].mean()

    print("fp = " + str(p))
    print("fq = " + str(q))
    print("fd = " + str(d))

    # for i in range(periodos):   
    model = SARIMAX(set_df, order=(p, d, q), seasonal_order=(p, d, q, 48))
    model_fit = model.fit()
    output = model_fit.forecast(periodos)
    yhat = output[0]
    model_future_predictions.append(yhat)
    set_df.append(yhat)

    print(set_df)
    # return model_future_predictions, set_df, model    
    return output, set_df, model

def split_dataframe(df, periodos):

    to_row = int(len(df[:-periodos]))
    training_set = list(df[0:to_row][kafsimo])
    testing_set = list(df[to_row:][kafsimo])  

    return training_set, testing_set  

df = pd.read_csv("Data/prices.csv", low_memory=False)

df['validation'] = df['validation'].str.replace(' ', '_')

df = df.loc[df['validation'] == nomos]

df = df[["price_date",
        kafsimo]]

df["price_date"] = pd.to_datetime(df["price_date"], dayfirst=True)

df = df.sort_values(by=['price_date'])

model_existing_predictions, training_set, modele = previous_prediction(df, periodos)

model_future_predictions, final_set, modelf = future_prediction(df, periodos)

df.set_index('price_date', inplace=True)

new_dates = [df.index[-1]+DateOffset(days=x) for x in range(1, periodos+1)]

df_pred = pd.DataFrame(new_dates, columns=['price_date'])
df_pred.set_index('price_date', inplace=True)
df_pred.drop(columns=[kafsimo])
df_pred[kafsimo+"_predictions"] = model_future_predictions
y = []

for i in range(periodos):
    y.append(i)

plt.scatter(y, model_future_predictions)
plt.show()

plt.figure(figsize=(12,5), dpi=100)
plt.plot(df[kafsimo], label='training')
# plt.plot(test, label='actual')
plt.plot(model_future_predictions, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# dh = df.dropna()
# dh = dh.set_index('price_date', inplace=True)
# dh = df.asfreq('Y')
# decomposition = sm.tsa.seasonal_decompose(dh[kafsimo], model='additive', 
#                             extrapolate_trend='freq') #additive or multiplicative is data specific
# fig = decomposition.plot()
# plt.show()