from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

app = Flask(__name__)
CORS(app)

# Global variables to store the loaded model
model_data = None
model = None
scaler = None
feature_columns = None

# JSON file to store predictions
PREDICTIONS_FILE = 'predictions.json'

def load_predictions():
    """Load existing predictions from JSON file"""
    if os.path.exists(PREDICTIONS_FILE):
        try:
            with open(PREDICTIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_prediction(prediction_data):
    """Save prediction to JSON file"""
    predictions = load_predictions()
    predictions.append(prediction_data)
    
    try:
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(predictions, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def load_model():
    """Load the trained model from pickle file"""
    global model_data, model, scaler, feature_columns
    
    try:
        with open('crypto_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        
        print("Model loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def home():
    """Serve the main HTML page"""
    return app.send_static_file('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making price predictions"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['open', 'high', 'low', 'volume', 'market_cap']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract features
        open_price = float(data['open'])
        high_price = float(data['high'])
        low_price = float(data['low'])
        volume = float(data['volume'])
        market_cap = float(data['market_cap'])
        
        # Calculate additional features (using default values for demo)
        # In a real scenario, you'd need historical data for these calculations
        price_change = 0.0  # Default value
        volume_change = 0.0  # Default value
        high_low_ratio = high_price / low_price if low_price > 0 else 1.0
        open_close_ratio = open_price / (open_price * 1.01)  # Assuming 1% increase as default
        
        # Create feature array
        features = [
            open_price, high_price, low_price, volume, market_cap,
            price_change, volume_change, high_low_ratio, open_close_ratio
        ]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Calculate confidence (simplified - using model's feature importance)
        confidence = np.mean(model.feature_importances_) * 100
        
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'input_features': {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'volume': volume,
                'market_cap': market_cap
            },
            'predicted_price': round(prediction, 2),
            'confidence': round(confidence, 2)
        }
        
        if save_prediction(prediction_data):
            print("Prediction saved successfully!")
        else:
            print("Failed to save prediction.")

        response = jsonify({
            'predicted_price': round(prediction, 2),
            'confidence': round(confidence, 2),
            'input_features': {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'volume': volume,
                'market_cap': market_cap
            },
            'timestamp': datetime.now().isoformat()
        })
        response.headers['Content-Type'] = 'application/json'
        return response
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """API endpoint to get model information"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        response = jsonify({
            'model_type': type(model).__name__,
            'feature_columns': feature_columns,
            'n_features': len(feature_columns),
            'model_loaded': True,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_.tolist()))
        })
        response.headers['Content-Type'] = 'application/json'
        return response
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/api/sample-data', methods=['GET'])
def sample_data():
    """API endpoint to get sample data for testing"""
    try:
        # Read the CSV file to get sample data
        df = pd.read_csv('bitcoin.csv')
        latest_data = df.iloc[-1]
        
        response = jsonify({
            'sample_data': {
                'open': float(latest_data['Open']),
                'high': float(latest_data['High']),
                'low': float(latest_data['Low']),
                'volume': float(latest_data['Volume']),
                'market_cap': float(latest_data['Market_Cap'])
            },
            'date': latest_data['Date']
        })
        response.headers['Content-Type'] = 'application/json'
        return response
        
    except Exception as e:
        return jsonify({'error': f'Failed to get sample data: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    response = jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to verify JSON responses"""
    response = jsonify({
        'message': 'Test endpoint working',
        'timestamp': datetime.now().isoformat()
    })
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """API endpoint to get all saved predictions"""
    try:
        print("Loading predictions from file...")
        predictions = load_predictions()
        print(f"Loaded {len(predictions)} predictions")
        
        response_data = {
            'predictions': predictions,
            'total_predictions': len(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Returning response: {response_data}")
        response = jsonify(response_data)
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        print(f"Error in get_predictions: {e}")
        return jsonify({'error': f'Failed to load predictions: {str(e)}'}), 500

@app.route('/api/predictions/clear', methods=['DELETE'])
def clear_predictions():
    """API endpoint to clear all saved predictions"""
    try:
        if os.path.exists(PREDICTIONS_FILE):
            os.remove(PREDICTIONS_FILE)
        response = jsonify({
            'message': 'All predictions cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        print(f"Error in clear_predictions: {e}")
        return jsonify({'error': f'Failed to clear predictions: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any unhandled exceptions"""
    print(f"Unhandled exception: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Flask API server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please run model_train.py first.") 