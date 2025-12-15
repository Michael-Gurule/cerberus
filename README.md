#  Real-Time Fraud Detection System

A production-ready machine learning system for detecting credit card fraud in real-time with sub-100ms latency. Built with XGBoost, FastAPI, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)

##  Business Problem

Financial institutions lose billions annually to credit card fraud, but aggressive fraud detection creates friction for legitimate customers. This system provides:

- **Real-time detection** with <100ms inference latency
- **High accuracy** with optimized precision-recall tradeoff
- **Business-optimized threshold** balancing fraud prevention vs customer experience

### Key Metrics

- **Fraud Detection Rate**: 89%+
- **False Alarm Rate**: <0.2%
- **Estimated Monthly Savings**: $450K+ (net benefit)
- **Average Inference Time**: <20ms

##  Features

- **Production ML Pipeline**: End-to-end workflow from data to deployment
- **FastAPI Service**: Real-time predictions with interactive documentation
- **Interactive Dashboard**: Streamlit app for monitoring and exploration
- **Business Metrics**: Cost-optimized decision threshold
- **Model Monitoring**: Performance tracking and feature importance

## Project Structure
```
fraud-detection-system/
├── data/
│   ├── generate_data.py          # Synthetic data generation
│   └── credit_card_transactions.csv
├── models/
│   ├── train_model.py             # Model training
│   └── fraud_detection_model.pkl
├── api/
│   └── api.py                     # FastAPI service
├── dashboard/
│   └── dashboard.py               # Streamlit dashboard
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup
```bash
# Clone repository
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data
```bash
python data/generate_data.py
```

### 3. Train Model
```bash
python models/train_model.py
```

### 4. Start API
```bash
uvicorn api.api:app --reload
```

API runs at `http://localhost:8000`
Documentation at `http://localhost:8000/docs`

### 5. Launch Dashboard
```bash
streamlit run dashboard/dashboard.py
```

Dashboard opens at `http://localhost:8501`

##  Model Performance

| Metric    | Value |
| --------- | ----- |
| Precision | 83%+  |
| Recall    | 89%+  |
| F1-Score  | 86%+  |
| ROC-AUC   | 0.967 |

### Business Impact (Monthly)

| Metric            | Value        |
| ----------------- | ------------ |
| Fraud Prevented   | $534,000     |
| Missed Fraud Cost | $57,500      |
| False Alarm Cost  | $2,300       |
| **Net Benefit**   | **$474,200** |

##  API Usage

### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_001",
    "customer_id": 1234,
    "amount": 250.00,
    "merchant_category": "online",
    "is_online": 1,
    "is_international": 0,
    "distance_from_home": 5.2,
    "transaction_hour": 14,
    "day_of_week": 2,
    "txn_count_1h": 0,
    "txn_count_24h": 2,
    "amount_sum_24h": 150.50,
    "customer_avg_amount": 85.30
  }'
```

### Response
```json
{
  "transaction_id": "TXN_001",
  "is_fraud": 0,
  "fraud_probability": 0.1234,
  "risk_level": "low",
  "inference_time_ms": 11.23,
  "timestamp": "2024-12-15T15:30:45.123456"
}
```

##  Technical Details

### Feature Engineering

- Transaction velocity (1h, 24h windows)
- Amount deviation from customer baseline
- Distance from home location
- Temporal patterns (hour, day of week)
- Merchant category (one-hot encoded)

### Model

- **Algorithm**: XGBoost Gradient Boosting
- **Optimization**: Custom cost function (fraud loss vs customer friction)
- **Class Imbalance**: Handled via scale_pos_weight
- **Threshold**: Optimized for business metrics

##  Testing
```bash
# Run tests
pytest tests/test_api.py -v

# Check inference latency
python -m pytest tests/test_api.py::test_inference_latency -v
```

## Future Improvements

- [ ] Add model retraining pipeline
- [ ] Implement A/B testing framework
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Add deep learning models (LSTM)
- [ ] Graph-based fraud detection

## Author

# Michael Gurule 
- GitHub: [@michael-gurule](https://github.com/michael-gurule)
- LinkedIn: [michaeljgurule](https://www.linkedin.com/in/michael-j-g-447aa2134/)
- Email: michaelgurule1164@gmail.com

