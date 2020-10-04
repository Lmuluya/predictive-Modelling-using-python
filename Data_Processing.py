import pandas as np

import pandas as pd

import os

os.getcwd()

## importing dataset 

data1 = pd.read_csv("C:\\Users\\mxp comps\\Desktop\\Educba\\Machine Learning Projects\\Projects on ML - Predictive Modeling with Python\\Data Preprocessing\\data.csv")

# Changing the working directory
os.chdir("C:\\Users\\mxp comps\\Desktop\\Educba\\Machine Learning Projects\\Projects on ML - Predictive Modeling with Python\\Data Preprocessing")

data2 = pd.read_csv("data.csv")

# importing a file
pd.DataFrame.to_csv(data2, "data5.csv")

# Definig the dependent and independent variables
data = pd.read_csv("Data.csv")

x = data.iloc[:,:-1].values

y = data.iloc[:,3].values