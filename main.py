import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

# Load the dataset
df = pd.read_csv('winequality-red.csv') # default reading

print(df.head()) # print the first 5 rows of the dataset