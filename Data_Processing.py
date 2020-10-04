import pandas as np

import pandas as pd

import os

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

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

# Data modeling
simp = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
simp = SimpleImputer().fit(x[:, 1:3]) 
x[:, 1:3] = simp.transform(x[:, 1:3])

# Encoding the cartegorical variables
label_encoder_y = LabelEncoder()

y = label_encoder_y.fit_transform(y)

# lable encoding based on countries
label_encoder_x = LabelEncoder()


x[:,0] = label_encoder_x.fit_transform(x[:,0])

# Creating dummy variables
onehot = OneHotEncoder(categorical_features = [0])
x = onehot.fit_transform(x).toarray()


