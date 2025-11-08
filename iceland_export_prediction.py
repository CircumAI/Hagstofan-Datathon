"""
Iceland Export Prediction Model - ADVANCED MSE/MAE Optimization
This script preprocesses export data, extracts advanced features, 
trains an ensemble of models, and evaluates performance.

TARGET: **Minimize MSE and MAE**

ADVANCED OPTIMIZATIONS:
1. Enhanced Features:
   - Exponential moving averages (EMA)
   - Year-over-year growth rates
   - Interaction features (EUR×CPI, USD×month)
2. Bidirectional LSTM:
   - Captures patterns from both past and future context
3. GRU Architecture:
   - Faster alternative that sometimes outperforms LSTM
4. Lightweight Ensemble:
   - BiLSTM (60%) + GRU (40%) weighted ensemble
5. Gradient Clipping:
   - Prevents exploding gradients (clipnorm=1.0)
6. Advanced Callbacks:
   - ReduceLROnPlateau for adaptive learning
   - Early stopping on MAE

Previous best: MSE: 55,318,506, MAE: 6,096
Target: Further reduction in both metrics!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class IcelandExportPredictor:
    """
    A class to predict Iceland exports using LSTM neural networks.
    """
    
    def __init__(self, window_size=6, epochs=300, batch_size=8):
        """
        Initialize the predictor optimized for low MSE/MAE.
        
        Args:
            window_size: Number of past time steps to use for prediction
            epochs: Number of training epochs
            batch_size: Smaller batch for better gradients
        """
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        
    def load_and_preprocess_data(self):
        """
        Load all data files and preprocess them into a unified dataset.
        
        Returns:
            DataFrame with preprocessed features
        """
        print("Loading data...")
        
        # 1. Load main export data
        df_export = pd.read_csv("The value of exports by month (2011-2025.csv", 
                               sep=";", skiprows=2)
        
        # Extract total exports (first row)
        export_cols = [col for col in df_export.columns if col.startswith('20')]
        export_row = df_export.iloc[0]
        
        # Create time series dataframe
        export_data = []
        for col in export_cols:
            if col != 'Unnamed: 0':
                try:
                    value = float(export_row[col].replace(',', '.'))
                    export_data.append({
                        'month': col,
                        'total_export': value
                    })
                except:
                    continue
        
        df = pd.DataFrame(export_data)
        df['date'] = pd.to_datetime(df['month'].str.replace('M', '-'), format='%Y-%m')
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} months of export data")
        
        # 2. Load and merge exchange rates
        try:
            df_exchange = pd.read_csv("Exchange-rates_2015-2025.csv", sep=";")
            df_exchange['date'] = pd.to_datetime(df_exchange['Dagsetning'], 
                                                 format='%d.%m.%Y', errors='coerce')
            
            # Convert exchange rate columns
            for col in ['Evra EUR miðgengi', 'Bandaríkjadalur USD miðgengi', 
                       'Sterlingspund GBP miðgengi']:
                df_exchange[col] = df_exchange[col].replace('-', np.nan)
                df_exchange[col] = pd.to_numeric(df_exchange[col].astype(str).str.replace(',', '.'), 
                                                errors='coerce')
            
            # Aggregate to monthly average
            df_exchange['year_month'] = df_exchange['date'].dt.to_period('M')
            monthly_exchange = df_exchange.groupby('year_month').agg({
                'Evra EUR miðgengi': 'mean',
                'Bandaríkjadalur USD miðgengi': 'mean',
                'Sterlingspund GBP miðgengi': 'mean'
            }).reset_index()
            monthly_exchange['date'] = monthly_exchange['year_month'].dt.to_timestamp()
            
            # Merge with main data
            df = df.merge(monthly_exchange[['date', 'Evra EUR miðgengi', 
                                           'Bandaríkjadalur USD miðgengi',
                                           'Sterlingspund GBP miðgengi']], 
                         on='date', how='left')
            
            # Forward fill missing exchange rates
            df['eur_rate'] = df['Evra EUR miðgengi'].fillna(method='ffill').fillna(method='bfill')
            df['usd_rate'] = df['Bandaríkjadalur USD miðgengi'].fillna(method='ffill').fillna(method='bfill')
            df['gbp_rate'] = df['Sterlingspund GBP miðgengi'].fillna(method='ffill').fillna(method='bfill')
            
            print(f"Added exchange rate features")
            
        except Exception as e:
            print(f"Warning: Could not load exchange rates: {e}")
            df['eur_rate'] = 140
            df['usd_rate'] = 130
            df['gbp_rate'] = 165
        
        # 3. Load and merge inflation data
        try:
            df_inflation = pd.read_csv("Inflation-Consumer price index.csv")
            df_inflation['date'] = pd.to_datetime(df_inflation['Month'].str.replace('M', '-'), 
                                                 format='%Y-%m')
            df_inflation.rename(columns={'Consumer price index Index': 'cpi'}, inplace=True)
            
            df = df.merge(df_inflation[['date', 'cpi']], on='date', how='left')
            df['cpi'] = df['cpi'].fillna(method='ffill').fillna(method='bfill')
            
            print(f"Added inflation (CPI) features")
            
        except Exception as e:
            print(f"Warning: Could not load inflation data: {e}")
            df['cpi'] = 500
        
        # 4. Create time-based features
        df['year'] = df['date'].dt.year
        df['month_num'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        
        # 5. Create lag features - more comprehensive
        for lag in [1, 2, 3, 6, 9, 12]:
            df[f'export_lag_{lag}'] = df['total_export'].shift(lag)
        
        # 6. Create rolling average features - more comprehensive
        for window in [3, 6, 9, 12]:
            df[f'export_ma_{window}'] = df['total_export'].rolling(window=window).mean()
            df[f'export_std_{window}'] = df['total_export'].rolling(window=window).std()
        
        # 7. Create exponential moving averages (better for recent trends)
        df['export_ema_3'] = df['total_export'].ewm(span=3, adjust=False).mean()
        df['export_ema_6'] = df['total_export'].ewm(span=6, adjust=False).mean()
        
        # 8. Create year-over-year growth rate
        df['export_yoy_growth'] = df['total_export'].pct_change(12)
        
        # 9. Create interaction features (economic indicators × temporal)
        df['eur_cpi_interaction'] = df['eur_rate'] * df['cpi']
        df['usd_month_interaction'] = df['usd_rate'] * df['month_num']
        
        # 10. Create trend features
        df['time_idx'] = np.arange(len(df))
        df['export_diff_1'] = df['total_export'].diff(1)
        df['export_diff_12'] = df['total_export'].diff(12)
        df['export_pct_change_1'] = df['total_export'].pct_change(1)
        df['export_pct_change_12'] = df['total_export'].pct_change(12)
        
        # Drop rows with NaN values (due to lag and rolling features)
        df_clean = df.dropna().reset_index(drop=True)
        
        print(f"Final dataset: {len(df_clean)} rows with {len(df_clean.columns)} features")
        print(f"Available features: {len([col for col in df_clean.columns if col not in ['date', 'month', 'year_month']])}")
        
        return df_clean
    
    def prepare_features(self, df):
        """
        Prepare feature matrix with advanced engineered features.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Tuple of (features, target)
        """
        # Enhanced feature set with new additions
        feature_cols = [
            'total_export',
            'eur_rate', 'usd_rate', 'gbp_rate',
            'cpi',
            'month_num',
            'month_sin', 'month_cos',
            # Lag features
            'export_lag_1', 'export_lag_2', 'export_lag_3', 'export_lag_6',
            # Rolling features
            'export_ma_3', 'export_ma_6', 'export_ma_12',
            'export_std_6',  # Volatility
            # NEW: Exponential moving averages
            'export_ema_3', 'export_ema_6',
            # NEW: Year-over-year growth
            'export_yoy_growth',
            # NEW: Interaction features
            'eur_cpi_interaction', 'usd_month_interaction',
            # Trends
            'export_diff_1'
        ]
        
        # Only keep features that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        print(f"\nUsing {len(feature_cols)} enhanced features")
        
        features = df[feature_cols].values
        target = df['total_export'].values
        
        return features, target
    
    def create_sequences(self, features, target):
        """
        Create time series sequences for LSTM.
        
        Args:
            features: Feature matrix
            target: Target values
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            y.append(target[i+self.window_size])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def build_model(self, input_shape, model_type='bilstm'):
        """
        Build enhanced model for lower MSE/MAE.
        
        Args:
            input_shape: Shape of input data (window_size, n_features)
            model_type: 'bilstm' or 'gru' for different architectures
            
        Returns:
            Compiled Keras model with Huber loss and gradient clipping
        """
        if model_type == 'gru':
            # GRU variant - faster and sometimes better
            model = keras.Sequential([
                GRU(128, return_sequences=True, input_shape=input_shape,
                    kernel_regularizer=keras.regularizers.l2(0.0005)),
                Dropout(0.2),
                GRU(64, return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(0.0005)),
                Dropout(0.2),
                GRU(32, kernel_regularizer=keras.regularizers.l2(0.0005)),
                Dropout(0.15),
                Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
                Dropout(0.1),
                Dense(16, activation='relu'),
                Dense(1)
            ])
        else:
            # Bidirectional LSTM variant (default)
            model = keras.Sequential([
                Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.0005))),
                Dropout(0.2),
                Bidirectional(LSTM(64, return_sequences=True,
                                  kernel_regularizer=keras.regularizers.l2(0.0005))),
                Dropout(0.2),
                LSTM(32, kernel_regularizer=keras.regularizers.l2(0.0005)),
                Dropout(0.15),
                Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
                Dropout(0.1),
                Dense(16, activation='relu'),
                Dense(1)
            ])
        
        # Use Huber loss with gradient clipping for stability
        optimizer = keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, model_type='bilstm'):
        """
        Train the model with specified architecture.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            model_type: 'bilstm' or 'gru'
            
        Returns:
            Training history
        """
        print(f"\nBuilding {model_type.upper()} model...")
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]), model_type=model_type)
        
        print(f"\nModel summary:")
        self.model.summary()
        
        # Callbacks - optimize for MAE
        early_stopping = EarlyStopping(
            monitor='val_mae',  # Monitor MAE instead of loss
            patience=40,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        
        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=15,
            min_lr=0.00001,
            verbose=1
        )
        
        print(f"\nTraining model for {self.epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate(self, y_test, y_pred):
        """
        Evaluate model performance.
        
        Args:
            y_test: Test targets (actual values)
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        print("\nEvaluating model...")
        
        # Flatten predictions if needed
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        epsilon = 1e-10
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100
        
        # Calculate accuracy (100 - MAPE)
        accuracy = max(0, 100 - mape)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Accuracy': accuracy
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:>15}: {value:>12.4f}")
        print("="*50)
        
        return metrics
    
    def plot_results(self, y_train, y_val, y_test, 
                    train_pred, val_pred, test_pred, history):
        """
        Plot training history and predictions.
        
        Args:
            y_train, y_val, y_test: Actual values
            train_pred, val_pred, test_pred: Predicted values
            history: Training history
        """
        # Plot training history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss During Training')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[0, 1].plot(history.history['mae'], label='Training MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE During Training')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predictions plot
        n_train = len(y_train)
        n_val = len(y_val)
        n_test = len(y_test)
        
        all_actual = np.concatenate([y_train, y_val, y_test])
        all_pred = np.concatenate([train_pred.flatten(), 
                                  val_pred.flatten(), 
                                  test_pred.flatten()])
        
        axes[1, 0].plot(all_actual, label='Actual', linewidth=2)
        axes[1, 0].plot(all_pred, label='Predicted', linewidth=2, alpha=0.7)
        axes[1, 0].axvline(x=n_train, color='r', linestyle='--', 
                          label='Train/Val Split', alpha=0.5)
        axes[1, 0].axvline(x=n_train+n_val, color='g', linestyle='--', 
                          label='Val/Test Split', alpha=0.5)
        axes[1, 0].set_title('Actual vs Predicted Export Values')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Export Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot for test set
        axes[1, 1].scatter(y_test, test_pred, alpha=0.6)
        axes[1, 1].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 
                       'r--', linewidth=2)
        axes[1, 1].set_title('Test Set: Actual vs Predicted')
        axes[1, 1].set_xlabel('Actual Export Value')
        axes[1, 1].set_ylabel('Predicted Export Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('export_prediction_results.png', dpi=300, bbox_inches='tight')
        print("\nPlots saved to 'export_prediction_results.png'")
        plt.show()

def main():
    """
    Main function with advanced optimizations for minimizing MSE and MAE.
    """
    print("\n" + "="*60)
    print("ICELAND EXPORT PREDICTION - ADVANCED OPTIMIZATION")
    print("="*60)
    
    # Initialize predictor optimized for low error
    predictor = IcelandExportPredictor(
        window_size=6,
        epochs=300,
        batch_size=8
    )
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data()
    
    # Prepare features
    features, target = predictor.prepare_features(df)
    
    # Normalize features using StandardScaler
    feature_scaler = StandardScaler()
    features_normalized = feature_scaler.fit_transform(features)
    
    # Normalize target using StandardScaler
    target_scaler = StandardScaler()
    target_normalized = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    predictor.feature_scaler = feature_scaler
    predictor.scaler = target_scaler
    
    # Create sequences
    X, y = predictor.create_sequences(features_normalized, target_normalized)
    
    # Split data: 80% train, 10% validation, 10% test (more training data)
    train_size = int(0.80 * len(X))
    val_size = int(0.10 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\nData split:")
    print(f"  Training:   {len(X_train)} sequences (80%)")
    print(f"  Validation: {len(X_val)} sequences (10%)")
    print(f"  Testing:    {len(X_test)} sequences (10%)")
    
    # Train lightweight ensemble: BiLSTM + GRU
    print("\n" + "="*60)
    print("TRAINING LIGHTWEIGHT ENSEMBLE (BiLSTM + GRU)")
    print("="*60)
    
    models_predictions = []
    
    for i, model_type in enumerate(['bilstm', 'gru']):
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Model ({i+1}/2)")
        print(f"{'='*60}")
        
        # Set seed for reproducibility
        np.random.seed(42 + i)
        tf.random.set_seed(42 + i)
        
        # Train model
        history = predictor.train(X_train, y_train, X_val, y_val, model_type=model_type)
        
        # Get predictions
        test_pred_norm = predictor.model.predict(X_test, verbose=0)
        
        # Denormalize
        test_pred = target_scaler.inverse_transform(test_pred_norm)
        models_predictions.append(test_pred)
    
    # Weighted ensemble: Give BiLSTM slightly more weight
    print("\n" + "="*60)
    print("CREATING WEIGHTED ENSEMBLE")
    print("="*60)
    
    # BiLSTM: 60%, GRU: 40%
    test_pred = 0.6 * models_predictions[0] + 0.4 * models_predictions[1]
    
    # Get training and validation predictions from best model (BiLSTM)
    np.random.seed(42)
    tf.random.set_seed(42)
    predictor.train(X_train, y_train, X_val, y_val, model_type='bilstm')
    train_pred_norm = predictor.model.predict(X_train, verbose=0)
    val_pred_norm = predictor.model.predict(X_val, verbose=0)
    train_pred = target_scaler.inverse_transform(train_pred_norm)
    val_pred = target_scaler.inverse_transform(val_pred_norm)
    
    # Denormalize targets
    y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_val_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Evaluate model on test set
    metrics = predictor.evaluate(y_test_actual, test_pred.flatten())
    
    # Plot results
    predictor.plot_results(y_train_actual, y_val_actual, y_test_actual,
                          train_pred, val_pred, test_pred, history)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Focus on MSE and MAE (not accuracy)
    print(f"\n🎯 MSE/MAE OPTIMIZATION RESULTS:")
    print(f"  MSE: {metrics['MSE']:,.0f}")
    print(f"  MAE: {metrics['MAE']:,.0f}")
    print(f"  RMSE: {metrics['RMSE']:,.0f}")
    
    if metrics['MAE'] < 6000:
        print(f"\n✓ EXCELLENT: MAE < 6,000 ISK!")
    elif metrics['MAE'] < 6500:
        print(f"\n✓ GOOD: MAE < 6,500 ISK!")
    else:
        print(f"\n→ Current MAE: {metrics['MAE']:.0f} ISK - Continue optimizing")
    
    return predictor, metrics

if __name__ == "__main__":
    predictor, metrics = main()
