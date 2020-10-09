# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the data set
dataset = pd.read_csv("C:\\Users\\mxp comps\\Desktop\\Educba\\Machine Learning Projects\\Projects on ML - Predictive Modeling with Python\\Linear Regression\\Salary_Data.csv")

# Selecting the data 
x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values

# Splitting the dataset into test and train dataset 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size = 1/3, random_state = 0)

# fitting a simple linear model to training set
from sklearn.linear_model import LinearRegression

# Creating an object
regressor = LinearRegression()

regressor.fit(x_train, y_train)

regressor.coef_

regressor.intercept_

# predicting test_set results
pred = regressor.predict(x_test)

# Calculating RMSE

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, pred))

rmse

# Visualisations for the training set
plt.scatter(x_train, y_train, color = 'blue')
plt.plot(x_train, regressor.predict(x_train), color ='green')
plt.title("Training Set(Salary Vs Experience)")
plt.xlabel("Experience in Years")
plt.ylabel("Salary")
plt.show()

# Visualisations for the testing set
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_train, regressor.predict(x_train), color ='green')
plt.title("Test Set(Salary Vs Experience)")
plt.xlabel("Experience in Years")
plt.ylabel("Salary")
plt.show()

# Combining both
plt.scatter(x_train, y_train, color = 'blue')
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color ='green')
plt.title("Salary Vs Experience")
plt.xlabel("Experience in Years")
plt.ylabel("Salary")
plt.show()


