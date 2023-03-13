import numpy as np
import joblib
from flask import Flask, jsonify, request

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Prepare the data for prediction
    year = data['year']
    month = data['month']
    X = np.array([[1, 1, year, month]])
    X_scaled = scaler.transform(X)

    # Make the prediction and return it
    prediction = model.predict(X_scaled)
    output = prediction[0][0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
