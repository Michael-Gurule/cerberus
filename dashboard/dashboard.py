"""
Streamlit dashboard for fraud detection monitoring
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Load model for local predictions
@st.cache_resource
def load_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, 'models', 'fraud_detection_model.pkl')
        
        from models.train_model import FraudDetectionModel
        model = FraudDetectionModel.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None

# Load sample data
@st.cache_data
def load_sample_data():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        csv_path = os.path.join(project_root, 'data', 'credit_card_transactions.csv')
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return None

def main():
    st.title("🔍 Real-Time Fraud Detection Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Make Prediction", "Model Performance", "Data Explorer"]
    )
    
    model = load_model()
    df = load_sample_data()
    
    if page == "Overview":
        show_overview(df)
    elif page == "Make Prediction":
        show_prediction_page(model)
    elif page == "Model Performance":
        show_performance_page(df, model)
    elif page == "Data Explorer":
        show_data_explorer(df)


def show_overview(df):
    """Overview page with key metrics"""
    st.header("📊 System Overview")
    
    if df is None:
        st.error("Data not loaded. Please run the data generation script first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Transactions",
            f"{len(df):,}",
            help="Total number of transactions in dataset"
        )
    
    with col2:
        fraud_count = df['is_fraud'].sum()
        fraud_rate = (fraud_count / len(df)) * 100
        st.metric(
            "Fraud Cases",
            f"{fraud_count:,}",
            f"{fraud_rate:.2f}%"
        )
    
    with col3:
        avg_amount = df['amount'].mean()
        st.metric(
            "Avg Transaction",
            f"${avg_amount:.2f}"
        )
    
    with col4:
        fraud_amount = df[df['is_fraud']==1]['amount'].sum()
        st.metric(
            "Fraud Amount",
            f"${fraud_amount:,.0f}"
        )
    
    st.markdown("---")
    
    # Time series of transactions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transactions Over Time")
        daily_txns = df.groupby(df['timestamp'].dt.date).size().reset_index()
        daily_txns.columns = ['date', 'count']
        
        fig = px.line(daily_txns, x='date', y='count', 
                     title='Daily Transaction Volume')
        fig.update_layout(xaxis_title="Date", yaxis_title="Transaction Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud Rate by Category")
        fraud_by_cat = df.groupby('merchant_category').agg({
            'is_fraud': ['sum', 'mean']
        }).reset_index()
        fraud_by_cat.columns = ['category', 'fraud_count', 'fraud_rate']
        fraud_by_cat['fraud_rate'] = fraud_by_cat['fraud_rate'] * 100
        
        fig = px.bar(fraud_by_cat, x='category', y='fraud_rate',
                    title='Fraud Rate by Merchant Category')
        fig.update_layout(xaxis_title="Category", yaxis_title="Fraud Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Transaction amount distribution
    st.subheader("Transaction Amount Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df[df['is_fraud']==0], x='amount', nbins=50,
                          title='Legitimate Transactions')
        fig.update_layout(xaxis_title="Amount ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df[df['is_fraud']==1], x='amount', nbins=50,
                          title='Fraudulent Transactions', color_discrete_sequence=['red'])
        fig.update_layout(xaxis_title="Amount ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)


def show_prediction_page(model):
    """Interactive prediction page"""
    st.header("🎯 Make a Prediction")
    
    if model is None:
        st.error("Model not loaded. Please train the model first.")
        return
    
    st.write("Enter transaction details to get a fraud prediction:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        transaction_id = st.text_input("Transaction ID", "TXN_TEST_001")
        customer_id = st.number_input("Customer ID", min_value=0, value=1234)
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=10.0)
        merchant_category = st.selectbox(
            "Merchant Category",
            ['grocery', 'gas', 'restaurant', 'retail', 'online', 'travel', 'entertainment', 'utilities']
        )
    
    with col2:
        is_online = st.selectbox("Transaction Type", ["In-Person", "Online"])
        is_international = st.selectbox("Location", ["Domestic", "International"])
        distance_from_home = st.number_input("Distance from Home (miles)", 
                                            min_value=0.0, value=5.0, step=1.0)
        transaction_hour = st.slider("Hour of Day", 0, 23, 14)
    
    with col3:
        day_of_week = st.selectbox("Day of Week", 
                                   ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                    "Friday", "Saturday", "Sunday"])
        txn_count_1h = st.number_input("Transactions (last 1h)", min_value=0, value=0)
        txn_count_24h = st.number_input("Transactions (last 24h)", min_value=0, value=2)
        customer_avg_amount = st.number_input("Customer Avg Amount", 
                                             min_value=0.0, value=75.0, step=5.0)
    
    if st.button("🔍 Predict Fraud", type="primary"):
        # Prepare transaction data
        day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                      "Friday": 4, "Saturday": 5, "Sunday": 6}
        
        amount_deviation = (amount - customer_avg_amount) / (customer_avg_amount + 1)
        amount_sum_24h = customer_avg_amount * txn_count_24h
        
        txn_data = pd.DataFrame([{
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'amount': amount,
            'merchant_category': merchant_category,
            'is_online': 1 if is_online == "Online" else 0,
            'is_international': 1 if is_international == "International" else 0,
            'distance_from_home': distance_from_home,
            'transaction_hour': transaction_hour,
            'day_of_week': day_mapping[day_of_week],
            'txn_count_1h': txn_count_1h,
            'txn_count_24h': txn_count_24h,
            'amount_sum_24h': amount_sum_24h,
            'amount_deviation': amount_deviation,
            'customer_avg_amount': customer_avg_amount
        }])
        
        # Make prediction
        pred_label, pred_proba = model.predict(txn_data)
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pred_label[0] == 1:
                st.error(f"⚠️ **FRAUD DETECTED**")
            else:
                st.success(f"✅ **LEGITIMATE**")
        
        with col2:
            st.metric("Fraud Probability", f"{pred_proba[0]*100:.1f}%")
        
        with col3:
            if pred_proba[0] < 0.3:
                risk = "🟢 Low"
            elif pred_proba[0] < 0.6:
                risk = "🟡 Medium"
            else:
                risk = "🔴 High"
            st.metric("Risk Level", risk)
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_proba[0]*100,
            title={'text': "Fraud Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': model.threshold * 100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)


def show_performance_page(df, model):
    """Model performance metrics"""
    st.header("📈 Model Performance")
    
    if model is None or df is None:
        st.error("Model or data not loaded.")
        return
    
    # Make predictions on full dataset
    with st.spinner("Generating predictions..."):
        pred_labels, pred_probas = model.predict(df)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df['is_fraud'], pred_labels)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Legitimate', 'Fraud'],
                       y=['Legitimate', 'Fraud'],
                       text_auto=True,
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Performance Metrics")
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'False Positive Rate'],
            'Value': [
                f"{precision*100:.1f}%",
                f"{recall*100:.1f}%",
                f"{f1*100:.1f}%",
                f"{fp/(fp+tn)*100:.2f}%"
            ]
        })
        st.table(metrics_df)
    
    # Probability distribution
    st.subheader("Fraud Score Distribution")
    df_viz = df.copy()
    df_viz['fraud_score'] = pred_probas
    
    fig = px.histogram(df_viz, x='fraud_score', color='is_fraud',
                      nbins=50, barmode='overlay',
                      labels={'is_fraud': 'Actual Label', 'fraud_score': 'Predicted Fraud Score'},
                      color_discrete_map={0: 'blue', 1: 'red'})
    st.plotly_chart(fig, use_container_width=True)


def show_data_explorer(df):
    """Interactive data explorer"""
    st.header("🔎 Data Explorer")
    
    if df is None:
        st.error("Data not loaded.")
        return
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fraud_filter = st.selectbox("Fraud Status", ["All", "Fraud Only", "Legitimate Only"])
    with col2:
        categories = ["All"] + df['merchant_category'].unique().tolist()
        category_filter = st.selectbox("Merchant Category", categories)
    with col3:
        min_amount = st.number_input("Min Amount", value=0.0)
    
    # Apply filters
    filtered_df = df.copy()
    
    if fraud_filter == "Fraud Only":
        filtered_df = filtered_df[filtered_df['is_fraud'] == 1]
    elif fraud_filter == "Legitimate Only":
        filtered_df = filtered_df[filtered_df['is_fraud'] == 0]
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['merchant_category'] == category_filter]
    
    filtered_df = filtered_df[filtered_df['amount'] >= min_amount]
    
    # Display data
    st.write(f"Showing {len(filtered_df):,} transactions")
    st.dataframe(
        filtered_df.head(1000),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Data",
        csv,
        "filtered_transactions.csv",
        "text/csv"
    )


if __name__ == "__main__":
    main()