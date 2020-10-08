# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the data set
dataset = pd.read_csv("C:\\Users\\mxp comps\\Desktop\\Educba\\Machine Learning Projects\\Projects on ML - Predictive Modeling with Python\\Linear Regression\\Salary_Data.csv")

# Selecting the data 
x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values
