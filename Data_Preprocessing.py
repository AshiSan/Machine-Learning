import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")

# y is the dependent variable (on which the prediction is based), X are the features (independent variables)
X = dataset.iloc[:, :-1].values   # iloc = index location[row, columns]
y = dataset.iloc[:, -1].values 
print(X)
print(y)

# Taking care of missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
print(X)

# Encoding categorical variables 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
print(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting Dataset into Training and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
# For some machine learning models we want to avoid that some features will dominate other features. So we want to standardize all the features to have them in the same range.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:, 3:])
X_test[:,3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)
