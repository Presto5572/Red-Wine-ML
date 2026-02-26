# Author: Colson Keim

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

# Scit-learn imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
dfW = pd.read_csv("/Users/colekeim/Downloads/Programming/Analytics Lab/Red Wine ML/winequality-red.csv") # default reading

# print(df.head())
# print(df.dtypes) 
# print(dfW.describe())

# Data splitting into train, validation, and test sets
df_train_val, df_test = train_test_split(dfW, test_size=0.2, random_state=25)
df_train, df_val = train_test_split(df_train_val, test_size=(15/85), random_state=25)

# Training data
X_train = df_train.drop("quality", axis=1)
y_train = df_train["quality"]

# Validation data
X_val = df_val.drop("quality", axis=1)
y_val = df_val["quality"]

# Test data
X_test = df_test.drop("quality", axis=1)
y_test = df_test["quality"]

feature_cols = X_train.select_dtypes(include=[np.number]).columns

# Column Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),

])

# preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols)
    ]
)

# Pipeline
pipe_main = Pipeline(steps=[
    ("preprocessor", preprocessor),
    # ("model", LinearRegression())
    ("model", RandomForestRegressor(n_estimators=100, random_state=25))
])

pipe_main.fit(X_train, y_train)

# Instead of calling .score again, just use the r2_score function with your y_hats
from sklearn.metrics import r2_score, mean_squared_error

# 1. Predict once
y_hat_train = pipe_main.predict(X_train)
y_hat_val = pipe_main.predict(X_val)

# 2. Calculate metrics using those predictions
train_r2 = r2_score(y_train, y_hat_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_hat_train))

val_r2 = r2_score(y_val, y_hat_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_hat_val))

# 3. Print
print(f"Training   | R2: {train_r2:.3f} | RMSE: {train_rmse:.3f}")
print(f"Validation | R2: {val_r2:.3f} | RMSE: {val_rmse:.3f}")
