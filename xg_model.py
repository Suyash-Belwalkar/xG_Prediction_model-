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
        self.rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42
        )
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        self.lr_model = LogisticRegression(
            C=0.1, max_iter=2000, solver='saga', random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_shot_data(self, data, training=False):
        features = data.copy()
        
        # Numeric features
        features['angle_rad'] = np.radians(features['angle'])
        features['distance_squared'] = features['shot_distance'] ** 2
        features['angle_distance'] = features['angle_rad'] * features['shot_distance']
        features['distance_inv'] = 1 / (features['shot_distance'] + 1)
        features['is_close_range'] = (features['shot_distance'] < 10).astype(int)
        
        # One-hot encode categorical variables
        for col in ['shot_type', 'defensive_pressure', 'assist_type']:
            if col in features.columns:
                features = pd.get_dummies(features, columns=[col], prefix=[col[:7]])
        
        # Interaction terms
        if 'defen_high' in features.columns:
            features['pressure_distance'] = features['defen_high'] * features['shot_distance']
            features['pressure_angle'] = features['defen_high'] * features['angle_rad']
        
        if 'goal' in features.columns:
            features.drop(columns=['goal'], inplace=True)

        if training:
            self.feature_columns = features.columns.tolist()

        return features
    
    def train(self, shots_data):
        X = self.prepare_shot_data(shots_data, training=True)
        y = shots_data['goal']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.rf_model.fit(X_train_scaled, y_train)
        self.xgb_model.fit(X_train_scaled, y_train)
        self.lr_model.fit(X_train_scaled, y_train)
        
        self.feature_columns = X.columns.tolist()
        self.save_model('xg_models.pkl')

        return {
            'random_forest': {
                'train_accuracy': float(self.rf_model.score(X_train_scaled, y_train)),
                'test_accuracy': float(self.rf_model.score(X_test_scaled, y_test))
            },
            'xgboost': {
                'train_accuracy': float(self.xgb_model.score(X_train_scaled, y_train)),
                'test_accuracy': float(self.xgb_model.score(X_test_scaled, y_test))
            },
            'logistic_regression': {
                'train_accuracy': float(self.lr_model.score(X_train_scaled, y_train)),
                'test_accuracy': float(self.lr_model.score(X_test_scaled, y_test))
            }
        }
    
    def predict_xg(self, shot_data):
        X = self.prepare_shot_data(shot_data)
        
        for feature in self.feature_columns:
            if feature not in X.columns:
                X[feature] = 0

        X = X[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        rf_predictions = self.rf_model.predict_proba(X_scaled)[:, 1]
        xgb_predictions = self.xgb_model.predict_proba(X_scaled)[:, 1]
        lr_predictions = self.lr_model.predict_proba(X_scaled)[:, 1]
        
        # Ensemble prediction
        ensemble_predictions = (0.4 * rf_predictions + 0.4 * xgb_predictions + 0.2 * lr_predictions)
        
        return {
            'random_forest': rf_predictions.tolist(),
            'xgboost': xgb_predictions.tolist(),
            'logistic_regression': lr_predictions.tolist(),
            'ensemble': ensemble_predictions.tolist()
        }
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'rf_model': self.rf_model,
                'xgb_model': self.xgb_model,
                'lr_model': self.lr_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            saved_models = pickle.load(f)
            self.rf_model = saved_models['rf_model']
            self.xgb_model = saved_models['xgb_model']
            self.lr_model = saved_models['lr_model']
            self.scaler = saved_models['scaler']
            self.feature_columns = saved_models['feature_columns']

xg_model = XGModel()

@app.route('/predict', methods=['POST'])
def predict():
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

@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify({"status": "success", "metrics": metrics})
    except FileNotFoundError:
        metrics = {
            'random_forest': {'train_accuracy': 0, 'test_accuracy': 0},
            'xgboost': {'train_accuracy': 0, 'test_accuracy': 0},
            'logistic_regression': {'train_accuracy': 0, 'test_accuracy': 0}
        }
        return jsonify({"status": "success", "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        shots_df = pd.DataFrame(data['shots'])
        results = xg_model.train(shots_df)
        
        with open('model_metrics.json', 'w') as f:
            json.dump(results, f)
            
        return jsonify({"status": "success", "metrics": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    try:
        xg_model.load_model('xg_models.pkl')
    except:
        print("No existing models found. Please train the models first.")
    
    app.run(debug=True)