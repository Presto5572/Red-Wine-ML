import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

set1 = "/Users/colekeim/Downloads/Programming/Analytics Lab/Red Wine ML/winequality-red.csv"
# Load the dataset
df = pd.read_csv(set1) # default reading

# print(df.head())
# print(df.dtypes) 
print(df.describe())

