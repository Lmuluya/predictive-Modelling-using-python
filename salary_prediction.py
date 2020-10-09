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



