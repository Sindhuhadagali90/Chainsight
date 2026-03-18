# ⛓️ ChainSight — Supply Chain Risk Intelligence v1.0

> A machine learning–powered dashboard for predicting global shipment delay risk in real time.

![Overview]
(https://github.com/gaganasindhu-ml/Chainsight/blob/ac87a5c3a69844b59f91618b7161fba4280b25dd/Screenshot/1.png)
---

## 📌 Overview

**ChainSight** is an end-to-end supply chain risk intelligence platform built on a dataset of **10,000 global shipments (2024–2025)**. It predicts whether a shipment will be delayed or on-time using an XGBoost classifier, and surfaces actionable risk insights through an interactive dashboard.

The project covers the full ML pipeline — from raw data ingestion and EDA through feature engineering, multi-model benchmarking, hyperparameter tuning, and deployment as a live interactive app.

---

## ✨ Features

| Module | Description |
|---|---|
| 📊 **Overview** | KPI cards, average delay by route type, delivery status donut chart |
| ⚡ **Risk Predictor** | Real-time delay risk prediction with risk factor breakdown |
| 〰️ **EDA Insights** | Delay distribution, feature correlations, disruption event frequency |
| 📋 **Model Comparison** | Accuracy & ROC-AUC benchmarks across 5 classifiers |

---

## 🗂️ Project Structure

```
chainsight/
│
├── Global_Supply_Chain_Risk_Prediction_Structured.ipynb   # Main ML pipeline notebook
├── Global_Supply_Chain_Delay___Disruption_Risk_Prediction_eda__1_.ipynb  # EDA notebook
├── xgb_supply_chain_model.pkl                             # Saved XGBoost model
├── README.md
└── screenshots/
    ├── overview.png
    ├── risk_predictor.png
    ├── eda_insights.png
    └── model_comparison.png
```

---

## 🧪 ML Pipeline

### 1. Data Loading
- 10,000 shipment records across 6 origin cities and 5 global routes
- Features include route type, transport mode, product category, lead times, geopolitical risk, weather severity, inflation rate, shipping cost, and order weight

### 2. Feature Engineering
- Parsed `Order_Date` into year, month, day, and day-of-week components
- Dropped `Order_ID` (non-informative)
- Created binary target: `Is_Delayed` (1 if `Delay_Days > 0`, else 0)
- Removed data-leaky columns not available at prediction time

### 3. Safe Features Used
```python
safe_cols = [
    'Origin_City', 'Destination_City', 'Route_Type', 'Transportation_Mode',
    'Product_Category', 'Base_Lead_Time_Days', 'Scheduled_Lead_Time_Days',
    'Geopolitical_Risk_Index', 'Weather_Severity_Index', 'Inflation_Rate_Pct',
    'Shipping_Cost_USD', 'Order_Weight_Kg',
    'Order_Year', 'Order_Month', 'Order_Day', 'Order_DayOfweek'
]
```

### 4. Preprocessing
- Label encoding for categorical columns
- 80/20 train-test split
- `StandardScaler` applied for distance-based models (LR, SVM, KNN)

### 5. Models Trained

| Model | Accuracy | ROC-AUC |
|---|---|---|
| 🥇 **XGBoost** | **92.4%** | **0.941** |
| Random Forest | 89.8% | 0.912 |
| SVM | 87.6% | 0.889 |
| Logistic Regression | 84.3% | 0.861 |
| KNN | 82.1% | 0.838 |

### 6. Final XGBoost Hyperparameters
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=7,   # handles class imbalance (87.5% on-time vs 12.5% delayed)
    eval_metric='logloss',
    random_state=42
)
```

---

## 📈 Key EDA Findings

- **Delay Rate:** 12.5% (1,247 of 10,000 shipments) — significant class imbalance (~1:7)
- **Max Observed Delay:** 20 days
- **Highest Delay Route:** Suez (avg 1.41 days), lowest: Intra-Asia (avg 0.72 days)
- **Top Correlated Feature:** `Actual_Lead_Time` (r = 0.82) — excluded as data-leaky
- **Most Common Disruption:** Port Congestion (820 events), followed by Geopolitical Conflict (312) and Extreme Weather (115)
- **Shipping Cost** has near-zero correlation with delay (r = 0.01)
- **Santos, BR** is the highest-risk origin city

---

## 🔍 XGBoost — Top Feature Importances

| Feature | Importance |
|---|---|
| Sched_Lead_Time | 0.31 |
| Base_Lead_Time | 0.22 |
| Route_Type | 0.14 |
| Origin_City | 0.09 |
| Weather_Index | 0.07 |
| Geopolitical_Idx | 0.06 |
| Transport_Mode | 0.05 |
| Product_Cat | 0.04 |
| Shipping_Cost | 0.01 |

---

## ⚡ Risk Predictor — How It Works

The Risk Predictor tab accepts the following inputs and returns a real-time delay probability:

| Input | Range / Options |
|---|---|
| Route Type | Pacific, Suez, Atlantic, Intra-Asia, Commodity |
| Transport Mode | Sea, Air, Road, Rail |
| Product Category | Textiles, Electronics, Machinery, etc. |
| Base Lead Time | Days (numeric) |
| Scheduled Lead Time | Days (numeric) |
| Geopolitical Risk | 0–1 |
| Weather Severity | 0–10 |
| Inflation Rate | % |
| Shipping Cost | USD |
| Order Weight | Kg |

The output includes:
- **Overall delay probability %** with a gauge meter
- **Risk level label** (Low / Moderate / High / Critical)
- **Risk Factor Breakdown** — individual contribution of each feature

---

## 🛠️ Tech Stack

- **ML:** Python, XGBoost, scikit-learn, pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** Streamlit (or equivalent interactive app framework)
- **Model Serialization:** joblib

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib
```

### Run the Notebooks
```bash
# EDA
jupyter notebook "Global_Supply_Chain_Delay___Disruption_Risk_Prediction_eda__1_.ipynb"

# Full ML pipeline
jupyter notebook "Global_Supply_Chain_Risk_Prediction_Structured.ipynb"
```

### Load the Saved Model
```python
import joblib

model = joblib.load('xgb_supply_chain_model.pkl')

# Example prediction
sample = [[...]]  # provide feature values in the correct order
prediction = model.predict(sample)
probability = model.predict_proba(sample)[:, 1]
```

---

## 📊 Dataset Summary

| Attribute | Value |
|---|---|
| Total Records | 10,000 |
| Time Period | 2024–2025 |
| Routes Covered | Pacific, Suez, Atlantic, Intra-Asia, Commodity |
| Origin Cities | 6 (incl. Santos BR as highest risk) |
| Delay Rate | 12.5% |
| Features Used | 16 (non-leaky predictors) |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

*Built with ❤️ as a supply chain risk intelligence capstone project.*
