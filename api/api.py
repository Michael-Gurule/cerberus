"""
FastAPI service for real-time fraud detection
Run with: uvicorn api:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
import time
from datetime import datetime
import os
import sys

# Add parent directory to path to import model class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection service",
    version="1.0.0"
)

# Load model at startup
model_artifacts = None
model = None
scaler = None
feature_columns = None
threshold = None

try:
    # Look for model in models directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, 'models', 'fraud_detection_model.pkl')
    
    model_artifacts = joblib.load(model_path)
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    feature_columns = model_artifacts['feature_columns']
    threshold = model_artifacts['threshold']
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("API will start but predictions will not work until model is trained.")


class Transaction(BaseModel):
    """Transaction input schema"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: int = Field(..., description="Customer identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: str = Field(..., description="Merchant category")
    is_online: int = Field(..., ge=0, le=1, description="1 if online, 0 if in-person")
    is_international: int = Field(..., ge=0, le=1, description="1 if international")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in miles")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    txn_count_1h: int = Field(0, ge=0, description="Transactions in last 1 hour")
    txn_count_24h: int = Field(0, ge=0, description="Transactions in last 24 hours")
    amount_sum_24h: float = Field(0.0, ge=0, description="Total amount in last 24 hours")
    customer_avg_amount: float = Field(..., gt=0, description="Customer average transaction")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Prediction output schema"""
    transaction_id: str
    is_fraud: int
    fraud_probability: float
    risk_level: str
    inference_time_ms: float
    timestamp: str


def prepare_transaction_features(txn: Transaction) -> pd.DataFrame:
    """Convert transaction to feature DataFrame"""
    # Calculate amount deviation
    amount_deviation = (txn.amount - txn.customer_avg_amount) / (txn.customer_avg_amount + 1)
    
    # Create base features
    features = {
        'amount': txn.amount,
        'is_online': txn.is_online,
        'is_international': txn.is_international,
        'distance_from_home': txn.distance_from_home,
        'transaction_hour': txn.transaction_hour,
        'day_of_week': txn.day_of_week,
        'txn_count_1h': txn.txn_count_1h,
        'txn_count_24h': txn.txn_count_24h,
        'amount_sum_24h': txn.amount_sum_24h,
        'amount_deviation': amount_deviation
    }
    
    # One-hot encode merchant category
    categories = ['grocery', 'gas', 'restaurant', 'retail', 'online', 'travel', 'entertainment', 'utilities']
    for cat in categories:
        features[f'category_{cat}'] = 1 if txn.merchant_category == cat else 0
    
    df = pd.DataFrame([features])
    
    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_columns]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Fraud Detection API",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_status": "loaded",
        "model_trained": model_artifacts['trained_date'],
        "threshold": threshold
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud probability for a single transaction
    
    Returns fraud prediction with probability and risk level
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Prepare features
        X = prepare_transaction_features(transaction)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        fraud_proba = model.predict_proba(X_scaled)[0, 1]
        is_fraud = int(fraud_proba >= threshold)
        
        # Determine risk level
        if fraud_proba < 0.3:
            risk_level = "low"
        elif fraud_proba < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=round(fraud_proba, 4),
            risk_level=risk_level,
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_fraud_batch(transactions: list[Transaction]):
    """
    Batch prediction endpoint for multiple transactions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limited to 1000 transactions")
    
    results = []
    for txn in transactions:
        result = await predict_fraud(txn)
        results.append(result)
    
    return {
        "batch_size": len(transactions),
        "predictions": results
    }


@app.get("/model/info")
async def model_info():
    """Get model metadata and performance info"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "trained_date": model_artifacts['trained_date'],
        "decision_threshold": threshold,
        "n_features": len(feature_columns),
        "feature_names": feature_columns[:10]  # Show first 10 features
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)