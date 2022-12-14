# https://analyticsindiamag.com/complete-guide-to-sarimax-in-python-for-time-series-modeling/  

import pandas as pd
import matplotlib.pyplot as plt

kafsimo = 'unleaded_95'
nomos = 'ΝΟΜΟΣ_ΔΩΔΕΚΑΝΗΣΟΥ'
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

# before the Interpolation
dm.plot()
plt.show()

#Interpolate in forward order across the column:
dm.interpolate(method ='linear', limit_direction ='forward', inplace=True)

# after the Interpolation
dm.plot()
plt.show()

################################################################################################
# check that to see if I can automate the p and q values

# order_aic_bic =[]
#     # Loop over AR order
# for p in range(8):
#     # Loop over MA order
#     for q in range(8):
#         # Fit models
#         model = SARIMAX(df['close'].dropna(), order=(p,d,q), trend='c')
#         results = model.fit()
#         # Add order and statistics to list
#         order_aic_bic.append((p, d, q, results.aic, results.bic))
    
# # Save parameters to a data frame 
# order_df = pd.DataFrame(order_aic_bic, columns=['p','d','q', 'aic', 'bic'])
    
# # Select the parameters with the lowest AIC 
# parameter_df = order_df.loc[(order_df['aic'] == order_df['aic'].min())]