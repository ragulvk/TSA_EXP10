<H1 ALIGN =CENTER> Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL...</H1>

### Date: 27-04-2024

### AIM :

To implement SARIMA model using python.

### ALGORITHM :

#### Step 1 :

Explore the dataset.

#### Step 2 :

Check for stationarity of time series.

#### Step 3 :

Determine SARIMA models parameters p, q.

#### Step 4 :

Fit the SARIMA model.

#### Step 5 :

Make time series predictions and Auto-fit the SARIMA model.

#### Step 6 :

Evaluate model predictions.

### PROGRAM :

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Temperature.csv')
data['date'] = pd.to_datetime(data['date'])

plt.plot(data['date'], data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.show()

def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['temp'])

plot_acf(data['temp'])
plt.show()
plot_pacf(data['temp'])
plt.show()

sarima_model = SARIMAX(data['temp'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['temp'][:train_size], data['temp'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```

### OUTPUT :

![img1](https://github.com/anto-richard/TSA_EXP10/assets/93427534/b11d65d5-9fcc-40cf-899b-e7cc88e484db)

![img2](https://github.com/anto-richard/TSA_EXP10/assets/93427534/0c50bd4d-eca9-432b-9cd7-cd65e9e1c87e)

![img3](https://github.com/anto-richard/TSA_EXP10/assets/93427534/d4670017-e174-481d-84ba-aa32bdbbdf02)

![img4](https://github.com/anto-richard/TSA_EXP10/assets/93427534/1780b679-49d2-4886-b942-dac79ed51e26)

### RESULT :

Thus, the program had ran successfully based on the SARIMA model using python.
