import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 


car_data = pd.read_csv("res\data.csv")

print(car_data.corr())

mask = np.triu(np.ones_like(car_data.corr()))
plt.figure(figsize=(13, 13))
sns.heatmap(car_data.corr(), annot=True, cmap = 'YlGnBu')




plt.figure(figsize=(20, 12))

plt.subplot(3, 4, 1) # 3 rows, 3 columns, 1st subplot = left
sns.boxplot(x='fueltype', y='price (USD)', data=car_data)

plt.subplot(3, 4, 2) # 3 rows, 3 columns, 2nd subplot = middle
sns.boxplot(x='aspiration', y='price (USD)', data=car_data)

plt.subplot(3, 4, 3) # 3 rows, 3 columns, 3rd subplot = right
sns.boxplot(x='doornumber', y='price (USD)', data=car_data)

plt.subplot(3, 4, 4)
sns.boxplot(x='carbody', y='price (USD)', data=car_data)

plt.subplot(3, 4, 5)
sns.boxplot(x='drivewheel', y='price (USD)', data=car_data)

plt.subplot(3, 4, 6)
sns.boxplot(x='enginelocation', y='price (USD)', data=car_data)


plt.subplot(3, 4, 7)
sns.boxplot(x='enginetype', y='price (USD)', data=car_data)

plt.subplot(3, 4, 8)
sns.boxplot(x='enginetype', y='price (USD)', data=car_data)

plt.subplot(3, 4, 9)
sns.boxplot(x='cylindernumber', y='price (USD)', data=car_data)

plt.subplot(3, 4, 10)
sns.boxplot(x='fuelsystem', y='price (USD)', data=car_data)



plt.show()