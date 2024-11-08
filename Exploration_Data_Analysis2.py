
import pandas as pd
import math
import time
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
import numpy as np
from math import sqrt
import random

cleanedDataFrame = pd.read_csv("DataFrame.csv")
cleanedDataFrame = cleanedDataFrame.drop(["txnHash", "age", "from", "to"], axis=1)

# Convert 'txnFee' and 'value' columns to numeric types, handling any potential formatting issues
cleanedDataFrame["txnFee"] = cleanedDataFrame["txnFee"].apply(lambda x: float(x))
pattern = r"(\d+\.?\d*)"
cleanedDataFrame["value"] = cleanedDataFrame["value"].apply(lambda x: float(re.findall(pattern, x)[0]))
numeric_cols = cleanedDataFrame.select_dtypes(include=['float64', 'int64']).columns

eda_summary = pd.DataFrame({
    "Column Names": cleanedDataFrame.columns,
    "Null Values": cleanedDataFrame.isnull().sum(),
    "Unique Values": cleanedDataFrame.nunique(),
    "Data Types": cleanedDataFrame.dtypes,
})

# Add mean, standard deviation, min, and max
eda_summary_numeric = pd.DataFrame({
    "Mean": cleanedDataFrame[numeric_cols].mean(),
    "Standard Deviation": cleanedDataFrame[numeric_cols].std(),
    "Min": cleanedDataFrame[numeric_cols].min(),
    "Max": cleanedDataFrame[numeric_cols].max(),
})

# Combine both summaries
eda_summary = pd.concat([eda_summary, eda_summary_numeric], axis=1)


print("First Few Rows of the Data:")
print(cleanedDataFrame.head())
print("\nEDA Summary:")
print(eda_summary)



# 1. Histogram of Numerical Features
cleanedDataFrame[numeric_cols].hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.suptitle('Histogram of Numerical Features')
plt.show()

# 2. Correlation Heatmap for Numerical Features
plt.figure(figsize=(10, 8))
sns.heatmap(cleanedDataFrame[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()