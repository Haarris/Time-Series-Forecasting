import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import scipy.stats as stats
import statsmodels.api as sm

df=pd.read_csv("calltotal.csv",parse_dates=["date"],index_col="date")

print(df.head())
print(df.dtypes)


df["date"]=pd.to_datetime(df["date"])
print(df.dtypes)

df.plot()
plt.show()

df.sort_index(inplace=True)

from statsmodels.tsa.seasonal import seasonal_decompose
decomp_results=seasonal_decompose(df["Total"])
decomp_results.plot()
plt.show()

from statsmodels .tsa.stattools import adfuller
results= adfuller(df["Total"])
print(results)


df_diff=df.Total.diff(1).dropna()
#df_diff=df.Total-df.Total.shift(3)
#df_diff=df_diff.dropna(inplace=False)


from statsmodels .tsa.stattools import adfuller
results= adfuller(df_diff)
print(results)


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
fig,(ax1,ax2)=plt.subplots(2,1)
plot_acf(df_diff,lags=20,zero=False,ax=ax1)
plot_pacf(df_diff,lags=20,zero=False,ax=ax2)
plt.show()


from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


from statsmodels.tsa.seasonal import seasonal_decompose
decomp_results=seasonal_decompose(df_diff)
decomp_results.plot()
plt.show()






######################



#######################

















order_aic_bic=[]
for p in range(10):
    for q in range(10):
        model=ARIMA(df.Total,order=(p,1,q))
        results=model.fit()
        order_aic_bic.append((p,q,results.aic,results.bic))

order_df=pd.DataFrame(order_aic_bic,columns=["p","q","aic","bic"])

print(order_df.sort_values("aic").head(5))


lowest_aic_row = order_df.iloc[order_df["aic"].idxmin()]
lowest_p, lowest_q = lowest_aic_row["p"], lowest_aic_row["q"]
print("p = {}, q = {}".format(int(lowest_p), int(lowest_q)))


#########################################

# Retrieve the rows with the three lowest AIC values
lowest_aic_rows = order_df.nsmallest(3, "aic")

# Loop over the rows and print the p and q values as integers
for _, row in lowest_aic_rows.iterrows():
    print("p = {}, q = {}".format(int(row["p"]), int(row["q"])))




first_p = int(order_df.sort_values("aic").iloc[0]["p"])
print(first_p)

second_p = int(order_df.sort_values("aic").iloc[1]["p"])
print(second_p)

third_p = int(order_df.sort_values("aic").iloc[2]["p"])
print(third_p)




first_q = int(order_df.sort_values("aic").iloc[0]["q"])
print(first_q)

first_q = int(order_df.sort_values("aic").iloc[1]["q"])
print(first_q)

first_q = int(order_df.sort_values("aic").iloc[2]["q"])
print(first_q)



###############################



a123=order_df.sort_values("aic").head(1)

print(a123.p)
p123=a123.p
q123=a123.q
#p123=14
#q123=18
r123=p123-q123
r123.reset_index(drop=True, inplace=True)


print(r123)
print(p123)
print(q123)

print(len(p123))


model=ARIMA(df.Total,order=(11,1,18))

results=model.fit()
print(results.summary())




results.plot_diagnostics()
plt.show()


forecast=results.get_prediction(start="2020-01-01",end="2023-02-12")
mean_forecast=forecast.predicted_mean



#print(confidence_interval)

plt.plot(df.index,df["Total"], label='observed')
plt.plot(mean_forecast.index,mean_forecast,label="forecast")
plt.xlabel('Date')
plt.ylabel('Total call volume')
plt.legend()
plt.show()

print(mean_forecast.tail(12))





##########
#############

S=7
data_diff=df_diff.diff().diff(S).dropna()



fig,(ax1,ax2)=plt.subplots(2,1)

ax1.set_title("Original")
df["Total"].plot(ax=ax1)
data_diff.plot(ax=ax2)
ax2.set_title("After S differencing")
plt.show()



from statsmodels.tsa.seasonal import seasonal_decompose
after_diff=seasonal_decompose(data_diff)
after_diff.plot()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
df1=seasonal_decompose(df["Total"])
df1.plot()
plt.show()




from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
fig,(ax1,ax2)=plt.subplots(2,1)

plot_acf(data_diff,lags=40,zero=False,ax=ax1)

plot_pacf(data_diff,lags=40,zero=False,ax=ax2)
plt.show()





from statsmodels.tsa.stattools import adfuller
results= adfuller(data_diff)
print(results)


import statsmodels.api as sm






order_aic_bic=[]
for P in range(10):
    for Q in range(10):
        model1=sm.tsa.statespace.SARIMAX(df.Total,order=(11,1,18), seasonal_order=(P,1,Q,7))
        results1=model1.fit()
        order_aic_bic.append((P,Q,results1.aic,results1.bic))

order_df=pd.DataFrame(order_aic_bic,columns=["P","Q","aic","bic"])
print(order_df.sort_values("aic").head(5))



model1=sm.tsa.statespace.SARIMAX(df.Total,order=(11,1,18), seasonal_order=(0,1,0,7))

results1=model1.fit()
#print(results1.summary())



results1.plot_diagnostics()
plt.show()




forecast1=results1.get_prediction(start="2020-01-01",end="2023-02-12")
mean_forecast1=forecast1.predicted_mean



confidence_interval1=forecast1.conf_int()


print(confidence_interval1.iloc[:,0])



print(mean_forecast1)

plt.plot(df.index,df["Total"], label='observed')
plt.plot(mean_forecast.index,mean_forecast,label="forecast")
plt.plot(mean_forecast1.index,mean_forecast1,label="+Seasonal forecast")
plt.fill_between(mean_forecast1.index,confidence_interval1.iloc[:,0],confidence_interval1.iloc[:,1],color="red",alpha=0.1)
plt.xlabel('Date')
plt.ylabel('Total call volume')
plt.legend()
plt.show()


print(mean_forecast1.tail(12))













print(len(mean_forecast1))



index1=mean_forecast1.index[1127:1139]
print(index1)



print(len(mean_forecast1))
forecasted_data1=mean_forecast1[1127:1139]
forecasted_data1= list(np.around(np.array(forecasted_data1),2))
print(forecasted_data1)


print(len(mean_forecast))
forecasted_data=mean_forecast[1127:1139]
forecasted_data= list(np.around(np.array(forecasted_data),2))
print(forecasted_data)


#orig_data=[16850,14353,11197,10202,15554,14677,15004,13366,12187,9926,9668,13988,13871,15021,14242,13728,11335,9976,15489,15317,15948,14236,12600,8571,7548,13179,15459,17319,16435,15341,13021]

orig_data=[17765,14243,13076,10843,9271,14969,13874,13214,12118,11386,9394,8555]








orig_data_high=[]
orig_data_low=[]
orig_data_high20=[]
orig_data_low20=[]

for i in range(12):
    orig_data_high.append (float(orig_data[i] *1.1 ))

orig_data_high= list(np.around(np.array(orig_data_high),2))
print(orig_data_high)



for i in range(12):
    orig_data_low.append (float(orig_data[i] *0.9 ))

orig_data_low= list(np.around(np.array(orig_data_low),2))
print(len(orig_data_low))
print(len(orig_data_high))



for i in range(12):
    orig_data_high20.append (float(orig_data[i] *1.2 ))

orig_data_high20= list(np.around(np.array(orig_data_high20),2))
print(orig_data_high20)



for i in range(12):
    orig_data_low20.append (float(orig_data[i] *0.8 ))

orig_data_low20= list(np.around(np.array(orig_data_low20),2))
print(len(orig_data_low20))
print(len(orig_data_high20))





print(len(orig_data))

forecasted_percentage=[]
for i in range(12):
   forecasted_percentage.append (float(forecasted_data1[i] - orig_data[i]) /orig_data[i] * 100 )


print(forecasted_percentage)


print(sum(forecasted_percentage))








confidence_interval2=forecast1.conf_int()

confidence_interval2=confidence_interval2[1127:1139]
print(confidence_interval2)

print(len(confidence_interval2.iloc[:,0]))









plt.plot(index1,orig_data, label='observed',linewidth=2)
plt.plot(index1,forecasted_data,label="forecast",linewidth=2)
plt.plot(index1,forecasted_data1,label="+Seasonal forecast",linewidth=2)
#plt.fill_between(index1,confidence_interval2.iloc[:,0],confidence_interval2.iloc[:,1],color="red",alpha=0.1)
plt.fill_between(index1,orig_data_high,orig_data_low,color="green",alpha=0.1)
plt.fill_between(index1,orig_data_high20,orig_data_low20,color="red",alpha=0.1)
plt.xlabel('Date')
plt.ylabel('Total call volume')
plt.legend()
plt.show()






def mean_absolute_error(y_true, y_pred):
    errors = [abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]
    return sum(errors) / len(y_true)


forecasted_data_a1_without_last_6 = forecasted_data[:-1]
orig_a1_without_last_6 = orig_data[:-1]

mae = mean_absolute_error(orig_a1_without_last_6,forecasted_data_a1_without_last_6)
print("Mean Absolute Error for Simple Forecast : ", mae)


mae1 = mean_absolute_error(orig_data,forecasted_data1)

print("Mean Absolute Error for Seasonal Forecast : ", mae1)






def mean_squared_error(y_true, y_pred):
    errors = [(y_true[i] - y_pred[i])**2 for i in range(len(y_true))]
    return sum(errors) / len(y_true)


mse = mean_squared_error(orig_data,forecasted_data)

print("Mean Squared Error for Simple Forecast : ", mse)


mse1 = mean_squared_error(orig_data,forecasted_data1)

print("Mean Squared Error for Seasonal Forecast : ", mse1)
























order_aic_bic=[]
for p in range(4):
    for q in range(4):
        model=ARIMA(df.Total,order=(p,1,q))
        results=model.fit()
        order_aic_bic.append((p,q,results.aic,results.bic))

order_df=pd.DataFrame(order_aic_bic,columns=["p","q","aic","bic"])


# Retrieve the rows with the three lowest AIC values
lowest_aic_rows = order_df.nsmallest(3, "aic")


# Loop over the rows and print the p and q values as integers
for _, row in lowest_aic_rows.iterrows():
    print("p = {}, q = {}".format(int(row["p"]), int(row["q"])))


first_p = int(order_df.sort_values("aic").iloc[0]["p"])
print(first_p)

second_p = int(order_df.sort_values("aic").iloc[1]["p"])
print(second_p)

third_p = int(order_df.sort_values("aic").iloc[2]["p"])
print(third_p)


first_q = int(order_df.sort_values("aic").iloc[0]["q"])
print(first_q)

second_q = int(order_df.sort_values("aic").iloc[1]["q"])
print(second_q)

third_q = int(order_df.sort_values("aic").iloc[2]["q"])
print(third_q)



##############################################################

last_date = df.index.max()
forecast_start = last_date + pd.DateOffset(days=1)
forecast_end = forecast_start + pd.DateOffset(days=5)
print(forecast_end)


# Run ARIMA model on data

model = ARIMA(df['Total'], order=(first_p,1,first_q))
results_a1 = model.fit()

forecast_a1 = results_a1.get_forecast(steps=6)
mean_forecast_a1 = forecast_a1.predicted_mean

forecast_a1_all = results_a1.get_forecast(steps=len(df.Total))
mean_forecast_a1_all = forecast_a1_all.predicted_mean




model = ARIMA(df['Total'], order=(second_p,1,second_q))
results_a2 = model.fit()

forecast_a2 = results_a2.get_forecast(steps=6)
mean_forecast_a2 = forecast_a2.predicted_mean

forecast_a2_all = results_a2.get_forecast(steps=len(df.Total))
mean_forecast_a2_all = forecast_a2_all.predicted_mean




model = ARIMA(df['Total'], order=(third_p,1,third_q))
results_a3 = model.fit()

forecast_a3 = results_a3.get_forecast(steps=6)
mean_forecast_a3 = forecast_a3.predicted_mean


forecast_a3_all = results_a3.get_forecast(steps=len(df.Total))
mean_forecast_a3_all = forecast_a3_all.predicted_mean



## SARIMAX
model1_s1=sm.tsa.statespace.SARIMAX(df.Total,order=(first_p,1,first_q), seasonal_order=(0,1,0,7))
results1_s1=model1_s1.fit()

forecast1_s1 = results1_s1.get_forecast(steps=6)
mean_forecast1_s1 = forecast1_s1.predicted_mean

forecast1_s1_all = results1_s1.get_forecast(steps=len(df.Total))
mean_forecast1_s1_all = forecast1_s1_all.predicted_mean



model1_s2=sm.tsa.statespace.SARIMAX(df.Total,order=(second_p,1,second_q), seasonal_order=(0,1,0,7))
results1_s2=model1_s2.fit()

forecast1_s2 = results1_s2.get_forecast(steps=6)
mean_forecast1_s2 = forecast1_s2.predicted_mean

forecast1_s2_all = results1_s2.get_forecast(steps=len(df.Total))
mean_forecast1_s2_all = forecast1_s2_all.predicted_mean




model1_s3=sm.tsa.statespace.SARIMAX(df.Total,order=(third_p,1,third_q), seasonal_order=(0,1,0,7))
results1_s3=model1_s3.fit()

forecast1_s3 = results1_s3.get_forecast(steps=6)
mean_forecast1_s3 = forecast1_s3.predicted_mean


forecast1_s3_all = results1_s3.get_forecast(steps=len(df.Total))
mean_forecast1_s3_all = forecast1_s3_all.predicted_mean




forecasted_data_a1 = mean_forecast_a1.loc[forecast_start:forecast_end].astype(int).tolist()
forecasted_data_a2 = mean_forecast_a2.loc[forecast_start:forecast_end].astype(int).tolist()
forecasted_data_a3 = mean_forecast_a3.loc[forecast_start:forecast_end].astype(int).tolist()
forecasted_data1_s1 = mean_forecast1_s1.loc[forecast_start:forecast_end].astype(int).tolist()
forecasted_data1_s2 = mean_forecast1_s2.loc[forecast_start:forecast_end].astype(int).tolist()
forecasted_data1_s3 = mean_forecast1_s3.loc[forecast_start:forecast_end].astype(int).tolist()


# MAE 

def mean_absolute_error(y_true, y_pred):
    errors = [abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]
    return sum(errors) / len(y_true)



mae1 = mean_absolute_error(df.Total, mean_forecast_a1_all)
mae1 = int(mae1)
print("Mean Absolute Error for first pair of p and q : ", mae1)

forecasted_data_a22 = forecasted_data_a2[:-6]
mae2 = mean_absolute_error(df.Total, mean_forecast_a2_all)
mae2 = int(mae2)
print("Mean Absolute Error for second pair of p and q : ", mae2)

forecasted_data_a33 = forecasted_data_a3[:-6]
mae3 = mean_absolute_error(df.Total, mean_forecast_a3_all)
mae3 = int(mae3)
print("Mean Absolute Error for third pair of p and q : ", mae3)





# Create a list of forecasts
forecasts = [forecasted_data_a1, forecasted_data_a2, forecasted_data_a3, forecasted_data1_s1, forecasted_data1_s2, forecasted_data1_s3, mae1, mae2, mae3]

print(forecasts)




















