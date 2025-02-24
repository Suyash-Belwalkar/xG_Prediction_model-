from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from flask_cors import CORS
import xgboost as xgb
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

class XGModel:
    def __init__(self):
        # Initialize all three models
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = None  # Store feature names used during training
        
    def prepare_shot_data(self, data, training=False):
        """
        Process raw shot data into features for the model.
        Expected columns: shot_distance, angle, body_part, etc.
        """
        features = data.copy()
        
        # Convert angle to radians
        if 'angle' in features.columns:
            features['angle_rad'] = np.radians(features['angle'])
        
        # Create interaction features
        features['distance_squared'] = features['shot_distance'] ** 2
        features['angle_distance'] = features['angle_rad'] * features['shot_distance']
        
        # One-hot encode categorical variables
        if 'body_part' in features.columns:
            features = pd.get_dummies(features, columns=['body_part'], prefix=['body_part'])
        
        # Drop non-feature columns
        if 'goal' in features.columns:
            features.drop(columns=['goal'], inplace=True)

        # If training, store feature names
        if training:
            self.feature_columns = features.columns.tolist()

        return features
    
    def train(self, shots_data):
        """Train all three xG models on historical shot data"""
        # Prepare features
        X = self.prepare_shot_data(shots_data, training=True)
        y = shots_data['goal']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train all three models
        self.rf_model.fit(X_train_scaled, y_train)
        self.xgb_model.fit(X_train_scaled, y_train)
        self.lr_model.fit(X_train_scaled, y_train)
        
        # Save feature names for future predictions
        self.feature_columns = X.columns.tolist()
        
        # Save trained models
        self.save_model('xg_models.pkl')

        # Return metrics for all models
        return {
            'random_forest': {
                'train_accuracy': self.rf_model.score(X_train_scaled, y_train),
                'test_accuracy': self.rf_model.score(X_test_scaled, y_test)
            },
            'xgboost': {
                'train_accuracy': self.xgb_model.score(X_train_scaled, y_train),
                'test_accuracy': self.xgb_model.score(X_test_scaled, y_test)
            },
            'logistic_regression': {
                'train_accuracy': self.lr_model.score(X_train_scaled, y_train),
                'test_accuracy': self.lr_model.score(X_test_scaled, y_test)
            }
        }
    
    def predict_xg(self, shot_data):
        """Predict xG for new shots using all three models"""
        X = self.prepare_shot_data(shot_data)
        
        for feature in self.feature_columns:
            if feature not in X.columns:
                X[feature] = 0  # Add missing feature with default value

        # Reorder columns to match training data
        X = X[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all three models
        rf_predictions = self.rf_model.predict_proba(X_scaled)[:, 1]
        xgb_predictions = self.xgb_model.predict_proba(X_scaled)[:, 1]
        lr_predictions = self.lr_model.predict_proba(X_scaled)[:, 1]
        
        # Return predictions from all models
        return {
            'random_forest': rf_predictions.tolist(),
            'xgboost': xgb_predictions.tolist(),
            'logistic_regression': lr_predictions.tolist()
        }
    
    def save_model(self, filepath):
        """Save the trained models to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'rf_model': self.rf_model,
                'xgb_model': self.xgb_model,
                'lr_model': self.lr_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
    
    def load_model(self, filepath):
        """Load trained models from file"""
        with open(filepath, 'rb') as f:
            saved_models = pickle.load(f)
            self.rf_model = saved_models['rf_model']
            self.xgb_model = saved_models['xgb_model']
            self.lr_model = saved_models['lr_model']
            self.scaler = saved_models['scaler']
            self.feature_columns = saved_models['feature_columns']

# Initialize model
xg_model = XGModel()

# API endpoints

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to get xG predictions for shots from all models"""
    try:
        data = request.json
        shots_df = pd.DataFrame(data['shots'])
        predictions = xg_model.predict_xg(shots_df)
        return jsonify({"status": "success", "predictions": predictions})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/')
def index():
    return send_file('index.html')


# Add this new endpoint to your Flask application

@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint to get model accuracy information"""
    try:
        # Check if model file exists and load metrics
        try:
            with open('model_metrics.json', 'r') as f:
                metrics = json.load(f)
        except FileNotFoundError:
            # If no metrics file exists yet, return empty metrics
            metrics = {
                'random_forest': {'train_accuracy': 0, 'test_accuracy': 0},
                'xgboost': {'train_accuracy': 0, 'test_accuracy': 0},
                'logistic_regression': {'train_accuracy': 0, 'test_accuracy': 0}
            }
        
        return jsonify({"status": "success", "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Then modify the train endpoint to save metrics to a file
@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint to train the models with new data"""
    try:
        data = request.json
        shots_df = pd.DataFrame(data['shots'])
        results = xg_model.train(shots_df)

        # Convert NumPy types to Python native types for JSON serialization
        serializable_results = {}
        for model_name, metrics in results.items():
            serializable_results[model_name] = {
                'train_accuracy': float(metrics['train_accuracy']),
                'test_accuracy': float(metrics['test_accuracy'])
            }
        
        # Save metrics to file for later retrieval
        with open('model_metrics.json', 'w') as f:
            json.dump(results, f)
            
        return jsonify({"status": "success", "metrics": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Try to load existing models
    try:
        xg_model.load_model('xg_models.pkl')
    except:
        print("No existing models found. Please train the models first.")
    
    app.run(debug=True)