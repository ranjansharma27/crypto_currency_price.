import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class CryptoPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = ['Open', 'High', 'Low', 'Volume', 'Market_Cap']
        self.target_column = 'Close'
        
    def load_data(self, file_path):
        """Load and preprocess the cryptocurrency data"""
        print("Loading data...")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Create additional features
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Remove rows with NaN values
        df = df.dropna()
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        print("Preparing features...")
        
        # Select features for training
        feature_cols = self.feature_columns + ['Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Open_Close_Ratio']
        X = df[feature_cols].values
        y = df[self.target_column].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Features prepared. X shape: {X_scaled.shape}, y shape: {y.shape}")
        return X_scaled, y
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        print("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model training completed!")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return X_test, y_test, y_pred
    
    def save_model(self, model_path='crypto_model.pkl'):
        """Save the trained model"""
        print(f"Saving model to {model_path}...")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns + ['Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Open_Close_Ratio']
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved successfully!")
    
    def predict_price(self, features):
        """Make a price prediction for given features"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        return prediction

def main():
    """Main function to train the model"""
    print("=== Cryptocurrency Price Prediction Model Training ===")
    
    # Initialize predictor
    predictor = CryptoPricePredictor()
    
    # Load data
    df = predictor.load_data('bitcoin.csv')
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train model
    X_test, y_test, y_pred = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model()
    
    print("\n=== Training Summary ===")
    print("Model trained and saved successfully!")
    print("You can now use the model for predictions via the API.")

if __name__ == "__main__":
    main() 