# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:09:59 2020

@author: LMULUYA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the data
dataset = pd.read_csv("C:\\Users\\mxp comps\\Desktop\\Educba\\Machine Learning Projects\\Projects on ML - Predictive Modeling with Python\\Logistic Regression\\Advertisement.csv")

# Going to use the age and estimated salary predict purchase of the product

# Determining independent variables
x = dataset.iloc[:,[2,3]].values
# Determining the dependent variable
y = dataset.iloc[:, 4].values

# Encoding the cartegorical variable
from sklearn.preprocessing import LabelEncoder

label_encoder_y = LabelEncoder()

y = label_encoder_y.fit_transform(y)

# Spliting the dataset betwee test set and test test 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,
                                                    random_state = 0)

# Features scalling 

from sklearn.preprocessing import StandardScaler

# Creating an empty standard scaler object
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)

# Fitting the logistic regression for the training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train,y_train)

# prdicting the model

y_pred = classifier.predict(x_test)

# Generating a confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

cm


#array([[65,  3],
#       [ 8, 24]], dtype=int64)

#TN = 65
#TP = 24
#FN = 8
#FP = 3

pd.crosstab(y_test,y_pred,rownames = ["True"],colnames =["Predicted"],
            margins = True)
