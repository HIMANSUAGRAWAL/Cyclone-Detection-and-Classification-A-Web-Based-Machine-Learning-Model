from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from project import classify_cyclone, cyclone_descriptions

app = Flask(__name__)

# Load model and features list from 'project.py'
rf, features = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    max_wind = float(request.form['max_wind'])
    min_pressure = float(request.form['min_pressure'])
    latitude = float(request.form['latitude'])
    year = int(request.form['year'])
    longitude = float(request.form['longitude'])

    # Ensure features are ordered correctly
    predicted_cyclone = classify_cyclone(max_wind, min_pressure, latitude, year, longitude)
    description = cyclone_descriptions.get(predicted_cyclone, "No description available for this cyclone type.")

    return jsonify({
        "cyclone_type": predicted_cyclone,
        "description": description
    })

if __name__ == '__main__':
    app.run(debug=True)
