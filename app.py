from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Pour autoriser le frontend externe (Bolt, React, etc.)

# Charger les modèles
model_conso = joblib.load("models/model_conso.pkl")
model_sol = joblib.load("models/model_sol.pkl")
model_eol = joblib.load("models/model_eol.pkl")

@app.route('/')
def home():
    return render_template('dashboard.html')  # ou "Bienvenue !"

# Endpoint IA consommation
@app.route('/api/predict/consommation', methods=['POST'])
def predict_conso():
    features = request.json['features']
    pred = float(model_conso.predict(np.array(features).reshape(1, -1))[0])
    return jsonify({'prediction': pred})

# Endpoint IA production solaire
@app.route('/api/predict/solaire', methods=['POST'])
def predict_sol():
    features = request.json['features']
    pred = float(model_sol.predict(np.array(features).reshape(1, -1))[0])
    return jsonify({'prediction': pred})

# Endpoint IA production éolienne
@app.route('/api/predict/eolienne', methods=['POST'])
def predict_eol():
    features = request.json['features']
    pred = float(model_eol.predict(np.array(features).reshape(1, -1))[0])
    return jsonify({'prediction': pred})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host='0.0.0.0', port=port)
