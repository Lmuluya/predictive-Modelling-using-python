# Importing the relevant libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sf


dataset = pd.read_csv("C:\\Users\\mxp comps\\Desktop\\Educba\\Machine Learning Projects\\Projects on ML - Predictive Modeling with Python\\Linear Regression\\tip.csv")

# Defining the variables
x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values 

# Fitting a reegression line
regressor = LinearRegression()

regressor.fit(x,y)

# Getting the coefficients 
regressor.coef_

regressor.intercept_

# Plotting the variables
x_mean = [np.mean(x) for i in x]

y_mean = [np.mean(y) for i in y]

plt.scatter(x,y)
plt.plot(x, regressor.predict(x), color = "red")
plt.plot(x, y_mean, linestyle = "--")
plt.plot(x_mean, y, linestyle = "--")
plt.title("Tip VS Bill")
plt.xlabel("Bill in $")
plt.ylabel("Tip in $")
plt.show()

# Summarised model

lm = sf.ols(formula = "Tip ~ Bill", data = dataset).fit()

lm.params

print(lm.summary())