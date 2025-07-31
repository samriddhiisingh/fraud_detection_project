import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Load your daily invoices file
df = pd.read_csv('synthetic_invoices.csv', parse_dates=['Date'])

# Convert Time column to proper format
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

print(" Data loaded:", df.shape)

# Optional: simple manual checks
print("\n Manual checks:")
invoice_counts = df['Invoice No'].value_counts()
duplicates = invoice_counts[invoice_counts > 1]
print(f"Duplicate invoices: {len(duplicates)}")

high_value_txns = df[df['Amount'] > 75000]
print(f"High-value transactions (>75k): {len(high_value_txns)}")

off_hour_txns = df[
    (df['Time'] < pd.to_datetime('06:00').time()) |
    (df['Time'] > pd.to_datetime('20:00').time())
]
print(f"Off-hour transactions: {len(off_hour_txns)}")

df['DayOfWeek'] = df['Date'].dt.dayofweek
weekend_txns = df[df['DayOfWeek'] >= 5]
print(f"Weekend transactions: {len(weekend_txns)}")

print("Manual checks done. Moving to smart anomaly detection...")

# Add engineered features
df['SecondsSinceMidnight'] = df['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

df = df.sort_values(['Vendor Name', 'Date'])
df['RollingVendorSpend'] = (
    df.groupby('Vendor Name')['Amount']
    .transform(lambda x: x.rolling(window=30, min_periods=1).sum())
)

vendor_avg = df.groupby('Vendor Name')['Amount'].transform('mean')
df['InvoiceDeviation'] = df['Amount'] / vendor_avg

# One-hot encode categorical columns
vendor_dummies = pd.get_dummies(df['Vendor Name'], prefix='Vendor', drop_first=True)
dept_dummies = pd.get_dummies(df['Department'], prefix='Dept', drop_first=True)

# Combine features for models
features = pd.concat([
    df[['Amount', 'DayOfWeek', 'SecondsSinceMidnight', 'RollingVendorSpend', 'InvoiceDeviation']],
    vendor_dummies,
    dept_dummies
], axis=1)

print("\n Features ready:", features.shape)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(features)
df['IF_Score'] = iso_forest.decision_function(features) * -1
df['Anomaly_IF'] = iso_forest.predict(features)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df['Anomaly_LOF'] = lof.fit_predict(features)
df['LOF_Score'] = lof.negative_outlier_factor_ * -1

# One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(features)
df['OCSVM_Score'] = ocsvm.decision_function(features) * -1
df['Anomaly_OCSVM'] = ocsvm.predict(features)

# Combine all scores into a single Fraud Risk Score
df['FraudRiskScore'] = df[['IF_Score', 'LOF_Score', 'OCSVM_Score']].mean(axis=1)
df['FraudRiskScore'] = 100 * (
    (df['FraudRiskScore'] - df['FraudRiskScore'].min()) /
    (df['FraudRiskScore'].max() - df['FraudRiskScore'].min())
)

print("\n Fraud Risk Score calculated!")
print(df[['Invoice No', 'Vendor Name', 'FraudRiskScore']].head())

# Save final results to output folder
df.to_csv('invoices_with_risk_score.csv', index=False)
print("\n Saved: invoices_with_risk_score.csv")
print(f"High risk invoices (>70): {len(df[df['FraudRiskScore'] > 70])}")
