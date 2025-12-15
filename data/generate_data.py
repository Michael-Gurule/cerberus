"""
Generate synthetic credit card transaction data for fraud detection
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_transactions(n_customers=10000, n_transactions=500000, fraud_rate=0.02):
    """
    Generate synthetic credit card transaction data
    
    Parameters:
    - n_customers: number of unique customers
    - n_transactions: total number of transactions
    - fraud_rate: proportion of fraudulent transactions
    """
    
    # Generate customer profiles
    customers = pd.DataFrame({
        'customer_id': range(n_customers),
        'home_lat': np.random.uniform(25, 48, n_customers),
        'home_lon': np.random.uniform(-125, -65, n_customers),
        'avg_transaction': np.random.lognormal(3.5, 0.8, n_customers),
        'typical_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_customers)
    })
    
    # Merchant categories
    categories = ['grocery', 'gas', 'restaurant', 'retail', 'online', 'travel', 'entertainment', 'utilities']
    
    # Generate transactions
    transactions = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(n_transactions):
        customer = customers.iloc[random.randint(0, n_customers-1)]
        is_fraud = random.random() < fraud_rate
        
        days_offset = random.randint(0, 364)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        timestamp = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)
        
        if is_fraud:
            amount = np.random.lognormal(5.0, 1.2)
            category = random.choice(categories)
            is_online = random.random() < 0.8
            is_international = random.random() < 0.4
            distance_from_home = np.random.uniform(500, 5000)
            hour = random.choice([0,1,2,3,4,22,23])
        else:
            amount = max(5, np.random.normal(customer['avg_transaction'], customer['avg_transaction']*0.4))
            category = customer['typical_category'] if random.random() < 0.7 else random.choice(categories)
            is_online = random.random() < 0.3
            is_international = random.random() < 0.05
            distance_from_home = np.random.exponential(10)
            
        transactions.append({
            'transaction_id': f'TXN_{i:08d}',
            'customer_id': customer['customer_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': category,
            'is_online': int(is_online),
            'is_international': int(is_international),
            'distance_from_home': round(distance_from_home, 2),
            'transaction_hour': hour,
            'day_of_week': timestamp.weekday(),
            'is_fraud': int(is_fraud)
        })
    
    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add velocity features
    df['txn_count_1h'] = 0
    df['txn_count_24h'] = 0
    df['amount_sum_24h'] = 0.0
    
    for idx, row in df.iterrows():
        customer_txns = df[(df['customer_id'] == row['customer_id']) & (df.index < idx)]
        
        if len(customer_txns) > 0:
            time_diff = (row['timestamp'] - customer_txns['timestamp']).dt.total_seconds() / 3600
            df.loc[idx, 'txn_count_1h'] = (time_diff <= 1).sum()
            df.loc[idx, 'txn_count_24h'] = (time_diff <= 24).sum()
            df.loc[idx, 'amount_sum_24h'] = customer_txns[time_diff <= 24]['amount'].sum()
    
    customer_avgs = df.groupby('customer_id')['amount'].mean().to_dict()
    df['customer_avg_amount'] = df['customer_id'].map(customer_avgs)
    df['amount_deviation'] = (df['amount'] - df['customer_avg_amount']) / (df['customer_avg_amount'] + 1)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic transaction data...")
    df = generate_transactions(n_customers=10000, n_transactions=500000, fraud_rate=0.02)
    
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'credit_card_transactions.csv')    
    df.to_csv(csv_path, index=False)

    print(f"\nDataset generated successfully!")
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nDataset saved to: {csv_path}")
