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

car_data = car_data.drop(['car_ID'], axis =1)
car_data = car_data.drop(['CarName'],axis=1)
car_data = car_data.drop(['CarManufacturer'], axis=1)

cars_numeric = car_data.select_dtypes(include =['int64','float64'])

categorical_cols = car_data.select_dtypes(include=['object'])


#car_dummies = pd.get_dummies(car_data[categorical_cols.columns])
#car_df = pd.concat([car_data, car_dummies], axis=1)
#car_df = car_df.drop(['CarManufacturer'], axis=1)

car_df = car_data

y = car_df.pop('price (USD)')
X = car_df



X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 100)



scaler = StandardScaler()

X_train= scaler.fit_transform(X_train)
X_test= scaler.fit_transform(X_test)


# train
lr = LinearRegression()
model = lr.fit(X_train, y_train)


# test
y_pred = model.predict(X_test)



#print(pd.DataFrame({'test':y_test,'pred':y_pred}).head())

r_squrt = r2_score(y_test, y_pred)
print('R^2 :',r_squrt)

mse = mean_squared_error(y_test,y_pred)
print('MSE :',mse)



# User Predict

manufIn = 'CarManufacturer_alfa-romero'
'''
carIn = pd.DataFrame.from_records([
    {
     'fueltype': 1, 
     'aspiration': 1,
     'doornumber': 1,
     'carbody': 1,
     'drivewheel': 3,
     'enginelocation': 1,
     'enginetype': 1,
     'cylindernumber':6,
     'enginesize':130,
     'fuelsystem': 6,
     'horsepower': 111,
     
     'CarManufacturer_Nissan':0,
     'CarManufacturer_alfa-romero':0,
     'CarManufacturer_audi':0,
     'CarManufacturer_bmw':0,
     'CarManufacturer_buick':0,
     'CarManufacturer_chevrolet':0,
     'CarManufacturer_dodge':0,
     'CarManufacturer_honda':0,
     'CarManufacturer_isuzu':0,
     'CarManufacturer_jaguar':0,
     'CarManufacturer_mazda':0,
     'CarManufacturer_mercury':0,
     'CarManufacturer_mitsubishi':0,
     'CarManufacturer_nissan':0,
     'CarManufacturer_peugeot':0,
     'CarManufacturer_plymouth':0,
     'CarManufacturer_porsche':0,
     'CarManufacturer_renault':0,
     'CarManufacturer_saab':0,
     'CarManufacturer_subaru':0,
     'CarManufacturer_toyota':0,
     'CarManufacturer_volkswagen':0,
     'CarManufacturer_volvo':0,
     'CarManufacturer_vw':0
     
     }
])
'''
# fill spec 
carIn=[(1,1,1,1,3,1,1,3,130,6,111)]

carIn = scaler.fit_transform(carIn)

predictPrice = model.predict(carIn)

print("Predict :",predictPrice[0])





