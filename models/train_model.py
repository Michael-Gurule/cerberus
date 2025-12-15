"""
Fraud Detection Model Training and Evaluation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_auc_score, average_precision_score
)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FraudDetectionModel:
    """
    Production-ready fraud detection model with business metrics
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.threshold = 0.5
        
    def prepare_features(self, df, is_training=True):
        """
        Engineer features for fraud detection
        """
        df = df.copy()
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Feature engineering
        feature_cols = [
            'amount', 'is_online', 'is_international', 'distance_from_home',
            'transaction_hour', 'day_of_week', 'txn_count_1h', 'txn_count_24h',
            'amount_sum_24h', 'amount_deviation'
        ]
        
        # One-hot encode merchant category
        category_dummies = pd.get_dummies(df['merchant_category'], prefix='category')
        
        # Combine features
        X = pd.concat([df[feature_cols], category_dummies], axis=1)
        
        if is_training:
            self.feature_columns = X.columns.tolist()
        else:
            # Ensure test data has same columns as training
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]
        
        return X
    
    def train(self, df, optimize_threshold=True):
        """
        Train fraud detection model
        """
        print("Preparing features...")
        X = self.prepare_features(df, is_training=True)
        y = df['is_fraud']
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train):,} transactions ({y_train.sum():,} fraud)")
        print(f"Validation set: {len(X_val):,} transactions ({y_val.sum():,} fraud)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train XGBoost model
        print("\nTraining XGBoost model...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        if optimize_threshold:
            print("\nOptimizing decision threshold...")
            self.threshold = self._optimize_threshold(y_val, y_pred_proba)
            print(f"Optimal threshold: {self.threshold:.3f}")
        
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Evaluation
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Fraud']))
        
        # Business metrics
        self._print_business_metrics(y_val, y_pred, y_pred_proba, X_val)
        
        # Feature importance
        self._plot_feature_importance()
        
        return X_val, y_val, y_pred_proba
    
    def _optimize_threshold(self, y_true, y_pred_proba):
        """
        Optimize threshold based on business cost function
        Assume: False Negative cost = $500, False Positive cost = $5
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Calculate cost for each threshold
        costs = []
        fn_cost = 500  # Cost of missing fraud
        fp_cost = 5    # Cost of false alarm (customer friction)
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            total_cost = (fn * fn_cost) + (fp * fp_cost)
            costs.append(total_cost)
        
        # Find threshold that minimizes cost
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold
    
    def _print_business_metrics(self, y_true, y_pred, y_pred_proba, X_val):
        """
        Calculate and print business-relevant metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Financial impact
        avg_fraud_amount = 500  # Assumed average fraud amount
        fn_cost = fn * avg_fraud_amount
        fp_cost = fp * 5  # Cost per false positive (customer friction)
        
        fraud_prevented = tp * avg_fraud_amount
        net_benefit = fraud_prevented - fn_cost - fp_cost
        
        print("\n" + "="*50)
        print("BUSINESS IMPACT METRICS")
        print("="*50)
        print(f"True Positives (Fraud Caught): {tp:,}")
        print(f"False Positives (Legitimate Flagged): {fp:,}")
        print(f"False Negatives (Fraud Missed): {fn:,}")
        print(f"True Negatives (Legitimate Passed): {tn:,}")
        print(f"\nFraud Detection Rate: {tp/(tp+fn)*100:.1f}%")
        print(f"False Alarm Rate: {fp/(fp+tn)*100:.2f}%")
        print(f"Precision: {tp/(tp+fp)*100:.1f}%")
        print(f"\nEstimated Monthly Impact (assuming validation set = 1 month):")
        print(f"  Fraud Prevented: ${fraud_prevented:,.0f}")
        print(f"  Cost of Missed Fraud: ${fn_cost:,.0f}")
        print(f"  Cost of False Alarms: ${fp_cost:,.0f}")
        print(f"  Net Benefit: ${net_benefit:,.0f}")
        
        # Model quality metrics
        print(f"\nModel Quality Metrics:")
        print(f"  ROC-AUC: {roc_auc_score(y_true, y_pred_proba):.3f}")
        print(f"  PR-AUC: {average_precision_score(y_true, y_pred_proba):.3f}")
    
    def _plot_feature_importance(self):
        """
        Plot feature importance
        """
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved to: feature_importance.png")
    
    def predict(self, df):
        """
        Make predictions on new data
        """
        X = self.prepare_features(df, is_training=False)
        X_scaled = self.scaler.transform(X)
        
        pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        pred_label = (pred_proba >= self.threshold).astype(int)
        
        return pred_label, pred_proba
    
    def save_model(self, path='fraud_detection_model.pkl'):
        """
        Save model and preprocessing artifacts
        """
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold,
            'trained_date': datetime.now().isoformat()
        }
        joblib.dump(model_artifacts, path)
        print(f"\nModel saved to: {path}")
    
    @classmethod
    def load_model(cls, path='fraud_detection_model.pkl'):
        """
        Load trained model
        """
        artifacts = joblib.load(path)
        
        model_obj = cls()
        model_obj.model = artifacts['model']
        model_obj.scaler = artifacts['scaler']
        model_obj.feature_columns = artifacts['feature_columns']
        model_obj.threshold = artifacts['threshold']
        
        print(f"Model loaded from: {path}")
        print(f"Model trained on: {artifacts['trained_date']}")
        
        return model_obj



if __name__ == "__main__":
    import os
    
    # Load data
    print("Loading transaction data...")
    
    # Get the correct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'credit_card_transactions.csv')
    
    print(f"Looking for CSV at: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at {csv_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in data folder: {os.listdir(os.path.join(project_root, 'data'))}")
        exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Train model
    fraud_model = FraudDetectionModel()
    X_val, y_val, y_pred_proba = fraud_model.train(df)

    #Save model to models directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'fraud_detection_model.pkl')
    fraud_model.save_model(model_path)
    
    print("\n" + "="*50)
    print("Model training complete!")
    print("="*50)
