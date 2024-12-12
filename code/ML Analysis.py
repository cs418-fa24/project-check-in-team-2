
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



cleanedDataFrame = pd.read_csv("DataFrame.csv")
cleanedDataFrame = cleanedDataFrame.drop(["txnHash", "age", "from", "to"], axis=1)
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


# 1. Feature Engineering

block_stats['rolling_avg_volume'] = block_stats['transaction_volume'].rolling(window=5).mean()
block_stats['volume_to_fee_ratio'] = block_stats['transaction_volume'] / (block_stats['avg_gas_fee'] + 1e-9)  # Avoid division by zero
block_stats.dropna(inplace=True)
features = block_stats[['avg_gas_fee', 'block', 'rolling_avg_volume', 'volume_to_fee_ratio']]
target = block_stats['transaction_volume']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 2. Model Training and Comparison


models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42)
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

results_df = pd.DataFrame(results).T
print("Model Performance Comparison:")
print(results_df)

# 3. Hyperparameter Tuning

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
print("Best Parameters for Random Forest:", grid_search.best_params_)

# 4. Evaluate the best model on test set
y_best_pred = best_rf_model.predict(X_test)
best_mae = mean_absolute_error(y_test, y_best_pred)
best_rmse = np.sqrt(mean_squared_error(y_test, y_best_pred))
best_r2 = r2_score(y_test, y_best_pred)

print("Best Random Forest Model Performance:")
print("MAE:", best_mae)
print("RMSE:", best_rmse)
print("R²:", best_r2)

# 5. Visualization

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Transaction Volume", color="blue")
plt.plot(y_best_pred, label="Best Model Predictions (Random Forest)", color="orange", linestyle='--')
plt.legend()
plt.title("Transaction Volume Prediction: Best Model vs Actual")
plt.xlabel("Test Set Index")
plt.ylabel("Transaction Volume")
plt.show()
