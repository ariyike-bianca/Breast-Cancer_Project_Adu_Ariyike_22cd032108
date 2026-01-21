from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('model/breast_cancer_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        features = [
            float(request.form['radius']),
            float(request.form['texture']),
            float(request.form['perimeter']),
            float(request.form['area']),
            float(request.form['concavity'])
        ]

        scaled_features = scaler.transform([features])
        result = model.predict(scaled_features)[0]

        prediction = "Benign" if result == 1 else "Malignant"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
