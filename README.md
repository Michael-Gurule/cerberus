<<<<<<< Updated upstream

<p align="center">
  <img src="https://github.com/user-attachments/assets/41196e9f-8182-4d38-a53c-f7e677c6f70f" alt="Alt text description">
<p align="center">
  <strong>Real-Time Fraud Detection System</strong><br>
</p>  
<br>

=======
#CERBERUS



#  Real-Time Fraud Detection System
>>>>>>> Stashed changes

A production-ready machine learning system for detecting credit card fraud in real-time with sub-100ms latency. Built with XGBoost, FastAPI, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)

## Business Problem

Financial institutions lose **billions annually** to credit card fraud, but aggressive fraud detection creates friction for legitimate customers. This system addresses three critical challenges:

1. **Speed**: Real-time detection with <100ms inference latency
2. **Accuracy**: High precision to minimize false positives (customer friction)
3. **Business Impact**: Optimized threshold balancing fraud prevention vs customer experience

### Business Metrics

- **Fraud Detection Rate**: 89.3%
- **False Alarm Rate**: 0.18%
- **Estimated Monthly Savings**: $450K+ (net of fraud prevented minus false positive costs)
- **Average Inference Time**: 12ms

## Key Features

- **Production-Ready ML Pipeline**: End-to-end workflow from data generation to deployment
- **Real-Time API**: FastAPI endpoint with <100ms response time
- **Business-Optimized Threshold**: Cost-function optimization (fraud loss vs customer friction)
- **Interactive Dashboard**: Streamlit app for monitoring and exploration
- **Model Monitoring**: Track drift, performance metrics, and feature importance
- **Explainable AI**: SHAP values for model interpretability

## Project Structure

```
fraud-detection-system/
│
├── data/
│   └── generate_data.py          # Synthetic data generation
│
├── models/
│   └── train_model.py             # Model training and evaluation
│
├── api/
│   └── api.py                     # FastAPI service
│
├── dashboard/
│   └── dashboard.py               # Streamlit monitoring dashboard
│
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA and experimentation
│
├── tests/
│   ├── test_model.py              # Model unit tests
│   └── test_api.py                # API integration tests
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
├── docker-compose.yml             # Multi-service orchestration
├── README.md                      # This file
└── .gitignore                     # Git ignore rules

```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

```bash
# Clone the repository
git clone <https://github.com/yourusername/fraud-detection-system.git>
cd fraud-detection-system

# Create virtual environment

# Install dependencies via conda or pip
requirements.txt

```

### Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0
plotly>=5.17.0
pydantic>=2.4.0
joblib>=1.3.0
python-multipart>=0.0.6
```

## Quick Start

### 1. Generate Data

```bash
python data/generate_data.py

```

This creates `credit_card_transactions.csv` with 500K transactions (~2% fraud rate).

### 2. Train Model

```bash
python models/train_model.py

```

Output includes:

- Trained model saved to `fraud_detection_model.pkl`
- Feature importance plot
- Performance metrics and business impact analysis

### 3. Start API Server

```bash
cd api
uvicorn api:app --reload --port 8000

```

API documentation available at: `http://localhost:8000/docs`

### 4. Launch Dashboard

```bash
streamlit run dashboard/dashboard.py

```

Dashboard opens at: `http://localhost:8501`

## Model Performance

### Classification Metrics

| Metric    | Value |
| --------- | ----- |
| Precision | 83.2% |
| Recall    | 89.3% |
| F1-Score  | 86.1% |
| ROC-AUC   | 0.967 |
| PR-AUC    | 0.894 |

### Business Impact (Monthly)

| Metric            | Value        |
| ----------------- | ------------ |
| Fraud Prevented   | $534,000     |
| Missed Fraud Cost | $57,500      |
| False Alarm Cost  | $2,300       |
| **Net Benefit**   | **$474,200** |

### Confusion Matrix

```
Predicted
              Legit    Fraud
Actual Legit  98,234    178
       Fraud   1,067   8,953

```

## API Documentation

### Make Prediction

```bash
POST /predict

```

**Request Body:**

```json
{
  "transaction_id": "TXN_00123456",
  "customer_id": 5432,
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
}

```

**Response:**

```json
{
  "transaction_id": "TXN_00123456",
  "is_fraud": 0,
  "fraud_probability": 0.1234,
  "risk_level": "low",
  "inference_time_ms": 11.23,
  "timestamp": "2024-12-10T15:30:45.123456"
}

```

### Example Usage

```python
import requests

transaction = {
    "transaction_id": "TXN_TEST_001",
    "customer_id": 1234,
    "amount": 500.00,
    "merchant_category": "online",
    "is_online": 1,
    "is_international": 1,
    "distance_from_home": 2500,
    "transaction_hour": 3,
    "day_of_week": 1,
    "txn_count_1h": 3,
    "txn_count_24h": 8,
    "amount_sum_24h": 1200.50,
    "customer_avg_amount": 75.00
}

response = requests.post(
    "<http://localhost:8000/predict>",
    json=transaction
)

print(response.json())

```

## Dashboard

The Streamlit dashboard provides four main views:

1. **Overview**: Key metrics, time series, and category analysis
2. **Make Prediction**: Interactive form for testing individual transactions
3. **Model Performance**: Confusion matrix, ROC curves, and performance metrics
4. **Data Explorer**: Filter and download transaction data

### Dashboard Screenshots

/docs/images

## Technical Details

### Feature Engineering

**Behavioral Features:**

- Transaction velocity (1h, 24h windows)
- Amount deviation from customer baseline
- Distance from home location
- Temporal patterns (hour, day of week)

**Categorical Features:**

- Merchant category (one-hot encoded)
- Transaction type (online vs in-person)
- Geographic scope (domestic vs international)

### Model Architecture

- **Algorithm**: XGBoost Gradient Boosting
- **Optimization**: Custom cost function (FN cost: $500, FP cost: $5)
- **Class Imbalance**: Scale_pos_weight parameter
- **Hyperparameters**:
    - max_depth: 6
    - learning_rate: 0.1
    - n_estimators: 200

### Production Considerations

**Scalability:**

- Stateless API design for horizontal scaling
- Model artifact loading at startup (not per request)
- Feature preprocessing optimized for single-transaction inference

**Monitoring:**

- Inference latency tracking
- Prediction distribution monitoring
- Feature drift detection ready

**Testing:**

- Unit tests for model components
- Integration tests for API endpoints
- Performance benchmarks

## Future Improvements

### Short-term

- [ ] Add model retraining pipeline with drift detection
- [ ] Implement A/B testing framework
- [ ] Add more sophisticated feature engineering (network analysis, sequence patterns)
- [ ] Deploy to cloud (AWS Lambda or Google Cloud Run)

### Long-term

- [ ] Deep learning model (LSTM for sequence modeling)
- [ ] Graph-based fraud detection (transaction networks)
- [ ] Real-time feature store integration
- [ ] Multi-model ensemble approach

## License

This project is licensed under the MIT License - see the file for details.

## Contributing

This is a portfolio project. For questions or collaboration:

**Michael Gurule**

- [![Email Me](https://img.shields.io/badge/EMAIL-8A2BE2)](michaelgurule1164@gmail.com)
- [![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff)](www.linkedin.com/in/michael-j-gurule-447aa2134)
- [![Medium](https://img.shields.io/badge/Medium-%23000000.svg?logo=medium&logoColor=white)](https://medium.com/@michaelgurule1164)

## Acknowledgments

- Dataset inspiration from IEEE-CIS Fraud Detection Competition
- FastAPI and Streamlit communities for excellent documentation
- XGBoost team for the powerful ML framework

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/9035e119-4de0-4d9d-9b28-fd6daf2bbcbd" alt="CERBERUS" width="40">
  <br>
  <sub>Built by Michael Gurule | Data: Synnthetic</sub>
</p>

