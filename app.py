from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load('model/flood_model.pkl')
# Fix XGBoost version compatibility: older pickled models missing this attribute
if not hasattr(model, 'use_label_encoder'):
    model.use_label_encoder = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['MonsoonIntensity']),
        float(request.form['TopographyDrainage']),
        float(request.form['RiverManagement']),
        float(request.form['Deforestation']),
        float(request.form['Urbanization']),
        float(request.form['ClimateChange']),
        float(request.form['DamsQuality']),
        float(request.form['Siltation']),
        float(request.form['AgriculturalPractices']),
        float(request.form['Encroachments']),
        float(request.form['IneffectiveDisasterPreparedness']),
        float(request.form['DrainageSystems']),
        float(request.form['CoastalVulnerability']),
        float(request.form['Landslides']),
        float(request.form['Watersheds']),
        float(request.form['DeterioratingInfrastructure']),
        float(request.form['PopulationScore']),
        float(request.form['WetlandLoss']),
        float(request.form['InadequatePlanning']),
        float(request.form['PoliticalFactors'])
    ]

    input_data = np.array([features])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "FLOOD LIKELY"
        color = "red"
        message = "High risk of flooding detected. Please take necessary precautions and follow emergency protocols immediately."
    else:
        result = "NO FLOOD"
        color = "green"
        message = "Low risk of flooding detected. Situation appears safe but continue monitoring weather conditions."

    return render_template('result.html',
                         result=result,
                         color=color,
                         message=message)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)