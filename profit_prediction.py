# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:54:55 2020

@author: LMULUYA
"""

# Multiple linear regression
# Importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset 
dataset = pd.read_csv("C:\\Users\\mxp comps\\Desktop\\Educba\\Machine Learning Projects\\Projects on ML - Predictive Modeling with Python\\Linear Regression\\Company Profit.csv")

# Taking profit as the dependent variable and the rest of the variables as indeoendent variables
# Spliting the data set
x = dataset.iloc[:,:-1].values

y = dataset.iloc[:, 4].values

# Including cartegorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label_encoder_x = LabelEncoder()

x[:, 3] = label_encoder_x.fit_transform(x[:,3])

onehotencoder_x = OneHotEncoder(categorical_features = [3])
x = onehotencoder_x.fit_transform(x).toarray()

x = x[:,1:]

# Spliting the dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,
                                                    random_state = 0)

# Fitting multiple linear reegression model to training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

regressor.intercept_ ## B_0

regressor.coef_

# Predicting the test set results

pred = regressor.predict(x_test)

# Calculationg RMSE - Root Mean Squared Error

from sklearn.metrics import mean_squared_error

from math import sqrt

rmse = sqrt(mean_squared_error(y_test,pred))

rmse

#####################################################

# Making an optimal model using stats model

# import stats model and add another variable 
import statsmodels.regression.linear_model as lm

x = np.append(arr=np.ones((50,1)).astype(int), values=x, axis=1)
X_opt=x[:,[0, 1, 2, 3, 4, 5]]
regressor_x=lm.OLS(endog=y, exog=X_opt).fit()
regressor_x.summary()

# Making an optimal model
# Remove variable with highest p-value (>0.05)

# R-squared:                       0.951
# Adj. R-squared:                  0.945

# Remove variable 2
X_opt=x[:,[0, 1, 3, 4, 5]]
regressor_x=lm.OLS(endog=y, exog=X_opt).fit()
regressor_x.summary()

# Remove 1

# R-squared:                       0.951
# Adj. R-squared:                  0.945

X_opt=x[:,[0, 3, 4, 5]]
regressor_x=lm.OLS(endog=y, exog=X_opt).fit()
regressor_x.summary()

# R-squared:                       0.951
# Adj. R-squared:                  0.945
# Remove 4
X_opt=x[:,[0, 3, 5]]
regressor_x=lm.OLS(endog=y, exog=X_opt).fit()
regressor_x.summary()

# R-squared:                       0.951
# Adj. R-squared:                  0.945
# Remove 5
X_opt=x[:,[0, 3]]
regressor_x=lm.OLS(endog=y, exog=X_opt).fit()
regressor_x.summary()

# Making model with significant variables

# Split
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.2,
                                                   random_state = 0)

regressor_2 = lm.OLS(endog = y_train, exog = x_train[:,[0,3]]).fit()
regressor_2.summary()




