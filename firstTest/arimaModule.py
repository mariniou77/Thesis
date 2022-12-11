# https://github.com/ggagnon1995/ARIMA_Algorithm/blob/main/ARIMA%20Algorithm.ipynb 

import pandas as pd
import sys
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
df = df[["record_date",
        kafsimo]]

# theto os index tin imerominia
# df.set_index('record_date', inplace=True)
df["record_date"] = pd.to_datetime(df["record_date"], dayfirst=True)

df = df.sort_values(by=['record_date'])

