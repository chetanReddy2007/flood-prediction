from flask import Flask, render_template, request
import joblib
import numpy as np
import xgboost as xgb
import os

# Prevent XGBoost OpenMP threading deadlock in uWSGI single-process mode
os.environ['OMP_NUM_THREADS'] = '1'

app = Flask(__name__)

# Feature names must match exactly what the model was trained with
FEATURE_NAMES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
    'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness',
    'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

# Lazy-load the model: load AFTER uWSGI forks the worker, not before.
# Loading at module level causes OpenMP to deadlock inside forked processes.
_booster = None

def get_booster():
    global _booster
    if _booster is None:
        _model = joblib.load(
            os.path.join(os.path.dirname(__file__), 'model', 'flood_model.pkl')
        )
        _booster = _model.get_booster()
        _booster.set_param({'nthread': 1})
    return _booster


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
    dmatrix = xgb.DMatrix(input_data, feature_names=FEATURE_NAMES)
    raw_pred = get_booster().predict(dmatrix)
    prediction = 1 if float(raw_pred[0]) > 0.5 else 0

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