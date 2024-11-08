
# Exploratory Data Analysis (EDA)


cleanedDataFrame = pd.read_csv("DataFrame.csv")
cleanedDataFrame = cleanedDataFrame.drop(["txnHash", "age", "from", "to"], axis=1)

# Convert 'txnFee' and 'value' columns to numeric types, handling any potential formatting issues
cleanedDataFrame["txnFee"] = cleanedDataFrame["txnFee"].apply(lambda x: float(x))
pattern = r"(\d+\.?\d*)"
cleanedDataFrame["value"] = cleanedDataFrame["value"].apply(lambda x: float(re.findall(pattern, x)[0]))

# 1. Transaction Fee Distribution
plt.figure(figsize=(10, 6))
sns.histplot(cleanedDataFrame["txnFee"], bins=30, kde=True)
plt.title("Distribution of Transaction Fees")
plt.xlabel("Transaction Fee (ETH)")
plt.ylabel("Frequency")
plt.show()

# 2. Block Number Correlation with Transaction Fee
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cleanedDataFrame, x="block", y="txnFee", alpha=0.5)
plt.title("Transaction Fee by Block Number")
plt.xlabel("Block Number")
plt.ylabel("Transaction Fee (ETH)")
plt.show()

# 3. Line plot of average transaction fee per block
# Group by block and calculate the mean transaction fee for each block
block_fee_avg = cleanedDataFrame.groupby("block")["txnFee"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=block_fee_avg, x="block", y="txnFee")
plt.title("Average Transaction Fee per Block")
plt.xlabel("Block Number")
plt.ylabel("Average Transaction Fee (ETH)")
plt.show()



