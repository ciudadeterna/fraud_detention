from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    return jsonify({'fraud_prob': prob})

if __name__ == '__main__':
    app.run()
