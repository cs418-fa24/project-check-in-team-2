!pip install selenium
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


# Data Cleaning


cleanedDataFrame = pd.read_csv("DataFrame.csv")
cleanedDataFrame = cleanedDataFrame.drop(["txnHash", "method", "age", "from", "to"], axis=1)



cleanedDataFrame["txnFee"] = cleanedDataFrame["txnFee"].apply(lambda x: float(x))
pattern = r"(\d+\.?\d*)"
cleanedDataFrame["value"] = cleanedDataFrame["value"].apply(lambda x: float(re.findall(pattern, x)[0]))


threshold = 3
columnsForProcess = ['value', 'txnFee']
outliersMask = ~((cleanedDataFrame[columnsForProcess] - cleanedDataFrame[columnsForProcess].mean()).abs() > threshold * cleanedDataFrame[columnsForProcess].std()).any(axis=1)
cleanedDataFrameWithoutOutlier = cleanedDataFrame[outliersMask]


valueMean = cleanedDataFrameWithoutOutlier['value'].mean()
valueStd = cleanedDataFrameWithoutOutlier['value'].std()
feeMean = cleanedDataFrameWithoutOutlier['txnFee'].mean()
feeStd = cleanedDataFrameWithoutOutlier['txnFee'].std()
print("Value Column:")
print("Mean:", valueMean)
print("Standard Deviation:", valueStd)
print("\ntxnFee Column:")
print("Mean:", feeMean)
print("Standard Deviation:", feeStd)



binValue=int(sqrt(len(cleanedDataFrameWithoutOutlier["value"])))
sns.distplot(cleanedDataFrameWithoutOutlier["value"],bins=binValue)
plt.title('Histogram of Value')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


removedZeroesDF = cleanedDataFrameWithoutOutlier["value"]
removedZeroesDF = removedZeroesDF[removedZeroesDF != 0]
removedZeroesDF = removedZeroesDF.reset_index()

binValue=int(sqrt(len(removedZeroesDF["value"])))
sns.distplot(removedZeroesDF["value"],bins=binValue)
plt.title('Histogram of Value (Zeros Removed)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

binValue=int(sqrt(len(cleanedDataFrameWithoutOutlier["txnFee"])))
sns.distplot(cleanedDataFrameWithoutOutlier["txnFee"],bins=binValue)
plt.title('Histogram of Fee')
plt.xlabel('Fee')
plt.ylabel('Frequency')
plt.show()



sns.violinplot(x=cleanedDataFrameWithoutOutlier["value"])
plt.xlabel("Value")
plt.title("Violin Plot of Value")
plt.show()

sns.violinplot(x=removedZeroesDF["value"])
plt.xlabel("Value")
plt.title("Violin Plot of Value (Zeros Removed)")
plt.show()

sns.violinplot(x=cleanedDataFrameWithoutOutlier["txnFee"])
plt.xlabel("Fee")
plt.title("Violin Plot of Fee")
plt.show()

sns.boxplot(x=cleanedDataFrameWithoutOutlier["value"])
plt.xlabel("Value")
plt.title("Box Plot of Value")
plt.show()

sns.boxplot(x=removedZeroesDF["value"])
plt.xlabel("Value")
plt.title("Box Plot of Value (Zeros Removed)")
plt.show()

sns.boxplot(x=cleanedDataFrameWithoutOutlier["txnFee"])
plt.xlabel("Fee")
plt.title("Box Plot of Fee")
plt.show()