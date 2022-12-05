# https://www.youtube.com/watch?v=pryXhOgDY9A

from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# kafsimo = str(sys.argv[1])
# nomos = str(sys.argv[2])
# periodos = int(sys.argv[3])

kafsimo = 'unleaded_95'
nomos = 'ΝΟΜΟΣ_ΑΡΤΗΣ'
periodos = 180

print(kafsimo)
print(nomos)
print(periodos)

def differencing_parameter(training_set):
    
    first_diff = adfuller(training_set)
    second_diff = adfuller(training_set)

    if ((first_diff[1] > 0.05) and (second_diff[1] > 0.05)): 
        d = 2
    elif((first_diff[1] > 0.05) and (second_diff[1] <= 0.05)):
        d =1 
    else: 
        d = 0 
        
    return d 

def previous_prediction(training_set, periodos):

    for i in range(n_test_obser):
        d = differencing_parameter(trainning_set)    
        model = ARIMA(trainning_set, order=(1, 2, d))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_existing_predictions.append(yhat)
        actual_test_value = testing_set[i]
        trainning_set.append(actual_test_value)

df = pd.read_csv("Data/prices.csv", low_memory=False)

df['validation'] = df['validation'].str.replace(' ', '_')

# vrisko oles tis egkrafes gia ton nomo pou dothike
df = df.loc[df['validation'] == nomos]

# kratao mono tis stiles pou thelo gia to modelo
df = df[["price_date",
        kafsimo]]

df["price_date"] = pd.to_datetime(df["price_date"], dayfirst=True)

# theto os index tin imerominia
# df.set_index('price_date', inplace=True)

df = df.sort_values(by=['price_date'])

print(df.head(15))
print(len(df))
print(int(len(df)*0.8))
# xwrizw to set mou se training_set kai testing_set
to_row = int(len(df)*0.8)
trainning_set = list(df[0:to_row][kafsimo])
testing_set = list(df[to_row:][kafsimo])

model_existing_predictions = []
model_future_predictions = []
n_test_obser = len(testing_set)

for i in range(n_test_obser):
    d = differencing_parameter(trainning_set)    
    model = ARIMA(trainning_set, order=(1, 2, d))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_future_predictions.append(yhat)
    actual_test_value = testing_set[i]
    trainning_set.append(actual_test_value) 

# print(output[0])       

print(model_fit.summary())       
print(model_future_predictions)
print(trainning_set[1024:])