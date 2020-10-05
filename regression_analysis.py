# Importing the relevant libraries 

import pandas as pd

import numpy as np

dataset = pd.read_csv("C:\\Users\\mxp comps\\Desktop\\Educba\\Machine Learning Projects\\Projects on ML - Predictive Modeling with Python\\Linear Regression\\tip.csv")

# Defining the variables
x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values 

