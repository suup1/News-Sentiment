import pandas as pd
from scipy.stats import ks_2samp
import mlflow
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load data
train = pd.read_csv("data/financial_news.csv", names=["label", "text"], encoding="latin1")

if not os.path.exists("logs/predictions.csv"):
    print("No predictions found. Skipping monitoring.")
    exit()

prod = pd.read_csv("logs/predictions.csv")

# Feature comparison (text length proxy)
train_len = train["text"].str.len()
prod_len = prod["text"].str.len()

stat, p_value = ks_2samp(train_len, prod_len)

print("Drift p-value:", p_value)

with mlflow.start_run(run_name="monitoring"):
    mlflow.log_metric("drift_p_value", p_value)

if p_value < 0.05:
    print("DATA DRIFT DETECTED")
else:
    print("No drift")