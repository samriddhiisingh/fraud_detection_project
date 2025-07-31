# Fraud Detection using Machine Learning

This project applies machine learning algorithms to detect potentially fraudulent vendor invoices using a combination of manual rules and anomaly detection models.

## Dataset
Synthetic invoice data with fields like:
- Vendor Name
- Amount
- Time
- Date
- Department

## Techniques Used
- Manual anomaly checks (duplicates, high-value, weekend/off-hour transactions)
- Feature engineering (rolling spend, time-of-day, deviation from average)
- Models:
  - Isolation Forest
  - Local Outlier Factor
  - One-Class SVM

## Output
Each invoice is assigned a `FraudRiskScore` between 0–100 based on combined anomaly scores.

## Tech Stack
- Python
- pandas, numpy
- scikit-learn

## Output
Final CSV file: `invoices_with_risk_score.csv`  
Includes risk scores and anomaly flags.

---


