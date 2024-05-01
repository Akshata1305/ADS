import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv(r'C:\Users\anike\Desktop\TCS\ADS\international-airline-passengers.csv')

# Convert the 'Month' column to datetime
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')

# Set 'Month' as the index
data.set_index('Month', inplace=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('International Airline Passengers')
plt.xlabel('Year')
plt.ylabel('Thousands of Passengers')
plt.grid(True)
plt.show()

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5,1,0)) # Example order, you might need to tune this
model_fit = model.fit()

# Forecast
forecast, stderr, conf_int = model_fit.forecast(steps=len(test))

# Plot the actual vs. forecasted values
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('International Airline Passengers Forecast')
plt.xlabel('Year')
plt.ylabel('Thousands of Passengers')
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print('Test RMSE:', rmse)
