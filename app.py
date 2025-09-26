from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "your_default_secret_key")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# --- Load Model ---
try:
    with open('Final_predictive_model/finalized_model.sav', 'rb') as file:
        model = pickle.load(file)
    print(f"Successfully loaded model {model}")
except FileNotFoundError:
    print("Error: 'finalized_model.sav' not found.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    raise

# --- Route to render the initial HTML page ---
@app.route('/')
def home():
    return render_template('predict.html')

# --- UNIFIED Prediction Endpoint (now returns JSON) ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json()
        features = [
            float(data.get('revolvingUtilization', 0)),
            float(data.get('age', 0)),
            float(data.get('late30To59Days', 0)),
            float(data.get('debtRatio', 0)),
            float(data.get('monthlyIncome', 0)),
            float(data.get('openCreditLines', 0)),
            float(data.get('late90PlusDays', 0)),
            float(data.get('realEstateLoans', 0)),
            float(data.get('late60To89Days', 0)),
            float(data.get('dependents', 0))
        ]
        
        final_features = [np.array(features)]
        prediction = model.predict(final_features)[0]
        probability_scores = model.predict_proba(final_features)
        probability_of_default = probability_scores[0][1] * 100
        
        # Send all results back as a JSON object
        return jsonify({
            'prediction': int(prediction),
            'probability': f"{probability_of_default:.2f}"
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Calculation error'}), 400


if __name__ == '__main__':
    print("Flask app is starting on http://127.0.0.1:8080")
    app.run(debug=True, host='127.0.0.1', port=8080)