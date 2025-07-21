from flask import Flask, render_template, request, flash, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

model = joblib.load("model/model.pkl")
encoders = joblib.load("model/encoders.pkl")

try:
    with open('model/brand_models.json', 'r') as f:
        brand_models = json.load(f)
except FileNotFoundError:
    print("Warning: brand_models.json not found. Run analyze_brand_models.py first.")
    brand_models = {}

def safe_transform(encoder, value, encoder_name):
    """Safely transform a value, handling unknown categories"""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        
        print(f"Warning: Unknown {encoder_name} '{value}'. Using default.")
        return encoder.transform([encoder.classes_[0]])[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None
    
    if request.method == 'POST':
        try:
            brand = safe_transform(encoders['Brand'], request.form['brand'], 'Brand')
            model_name = safe_transform(encoders['model'], request.form['model'], 'Model')
            year = int(request.form['year'])
            age = int(request.form['age'])
            km_driven = int(request.form['kmDriven'])
            transmission = safe_transform(encoders['Transmission'], request.form['transmission'], 'Transmission')
            owner = safe_transform(encoders['Owner'], request.form['owner'], 'Owner')
            fuel = safe_transform(encoders['FuelType'], request.form['fuel'], 'FuelType')

            data = np.array([[brand, model_name, year, age, km_driven, transmission, owner, fuel]])
            prediction = model.predict(data)[0]
            
        except Exception as e:
            error_message = f"Error making prediction: {str(e)}"
            print(f"Prediction error: {e}")

    
    available_options = {
        'brands': sorted(encoders['Brand'].classes_.tolist()),
        'models': sorted(encoders['model'].classes_.tolist()),
        'transmissions': sorted(encoders['Transmission'].classes_.tolist()),
        'owners': sorted(encoders['Owner'].classes_.tolist()),
        'fuels': sorted(encoders['FuelType'].classes_.tolist())
    }

    return render_template('index.html', 
                         prediction=prediction, 
                         error_message=error_message,
                         options=available_options,
                         brand_models=brand_models)

@app.route('/get_models/<brand>')
def get_models(brand):
    """API endpoint to get models for a specific brand"""
    models = brand_models.get(brand, [])
    return jsonify(models)

if __name__ == '__main__':
    app.run(debug=True)