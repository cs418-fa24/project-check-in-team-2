from bs4 import BeautifulSoup as bs
import requests as req
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

block_stats = cleanedDataFrame.groupby('block').agg(
    avg_gas_fee=('txnFee', 'mean'),   # Average transaction fee as a proxy for gas fee
    transaction_volume=('txnFee', 'size')  # Transaction count per block as volume
).reset_index()

volume_threshold_high = block_stats['transaction_volume'].quantile(0.75)
volume_threshold_low = block_stats['transaction_volume'].quantile(0.25)

# Categorize blocks based on transaction volume
block_stats['traffic_period'] = np.where(
    block_stats['transaction_volume'] >= volume_threshold_high, 'Peak',
    np.where(block_stats['transaction_volume'] <= volume_threshold_low, 'Low', 'Normal')
)

# Visualization: Scatter plot with different colors for peak and low-traffic periods
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=block_stats, x='transaction_volume', y='avg_gas_fee', hue='traffic_period',
    palette={'Peak': 'red', 'Low': 'blue', 'Normal': 'gray'}, alpha=0.6
)
plt.title("Relationship Between Transaction Volume and Average Gas Fees with Traffic Periods")
plt.xlabel("Transaction Volume (Transactions per Block)")
plt.ylabel("Average Gas Fee (ETH)")
plt.legend(title="Traffic Period")
plt.show()
