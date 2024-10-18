## DEVELOPED BY: VINOD KUMAR S
## REGISTER NO: 212222240116
## DATE:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

## AIM:
To Implement an Auto Regressive Model using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM
```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file (adjusting for the new columns)
data = pd.read_csv('coffeesales.csv', index_col='datetime', parse_dates=True)

# Display the first few rows (GIVEN DATA)
print("GIVEN DATA:")
print(data.head())

# Perform Augmented Dickey-Fuller test for stationarity on the 'money' column
result = adfuller(data['money'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit an AutoRegressive (AR) model with 13 lags
model = AutoReg(train['money'], lags=13)
model_fit = model.fit()

# Make predictions using the AR model
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test['money'], predictions)
print('Mean Squared Error:', mse)

# Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plt.figure(figsize=(10,6))
plt.subplot(211)
plot_pacf(train['money'], lags=13, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")
plt.subplot(212)
plot_acf(train['money'], lags=13, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")
plt.tight_layout()
plt.show()

# PREDICTION
print("PREDICTION:")
print(predictions)

# Plot the test data and predictions (FINAL PREDICTION)
plt.figure(figsize=(10,6))
plt.plot(test.index, test['money'], label='Actual Money')
plt.plot(test.index, predictions, color='red', label='Predicted Money')
plt.title('Test Data vs Predictions (FINAL PREDICTION)')
plt.xlabel('Date')
plt.ylabel('Money')
plt.legend()
plt.show()

```
## OUTPUT:

### GIVEN DATA
![image](https://github.com/user-attachments/assets/78360ecd-b033-4621-8d85-3270e6526b46)


### ADF-STATISTIC P-VALUE and MSE VALUE

![image](https://github.com/user-attachments/assets/d62b9206-5cc7-471e-846d-1a42b9844a7b)


### PACF - ACF

![image](https://github.com/user-attachments/assets/aade8210-8225-4bc8-bc16-e6b71a4f5132)


### PREDICTION

![image](https://github.com/user-attachments/assets/50c70931-0d6c-4e04-b0fc-1b90bbae2d8c)


### FINAL PREDICTION

![image](https://github.com/user-attachments/assets/8dee9d13-8092-4acd-b55f-7bb68ce3b203)


### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
