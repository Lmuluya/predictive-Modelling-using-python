# Import Packages

import numpy as np

import pandas as pd

import os

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

# Check current working directory

os.getcwd()

# Change WD

os.chdir("C:\\Users\\rec_001\\Desktop\\Predictive Modeling with Python\\Data Preprocessing")

# Import Dataset

data1 = pd.read_csv("C:\\Users\\rec_001\\Desktop\\Predictive Modeling with Python\\Data Preprocessing\\Data.csv")


data2 = pd.read_csv("Data.csv")



# Export dataset

pd.DataFrame.to_csv(data2, "data4.csv")


#########################################


# Data Preprocessing

data = pd.read_csv("Data.csv")

# df[Rows , Columns]

x = data.iloc[ : , :-1].values

y = data.iloc[ : , 3].values



# Handling missing values in data

impute = Imputer(missing_values="NaN", strategy="mean", axis=0)

impute = impute.fit(x[:,1:3])

x[:,1:3] = impute.fit_transform(x[:,1:3])


# Encode categorical variables

label_encoder_y = LabelEncoder()

y = label_encoder_y.fit_transform(y)



label_encoder_x = LabelEncoder()

x[:,0] = label_encoder_x.fit_transform(x[:,0])


# Create dummies

onehot = OneHotEncoder(categorical_features = [0])
x = onehot.fit_transform(x).toarray()


# Splitting dataset into training set and test set

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,
                                                    random_state = 0)



































