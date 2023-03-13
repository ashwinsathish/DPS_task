import json
import numpy as np
import os
import joblib
from keras.models import load_model
from azureml.core.model import Model

def init():
    global model, scaler
    model_path = Model.get_model_path('pred_model.h5')
    model = load_model(model_path)
    scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'scaler.pkl')
    scaler = joblib.load(scaler_path)

def run(raw_data):
    data = json.loads(raw_data)
    year = data['year']
    month = data['month']
    new_data = np.array([[1, 1, year, month]])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    return json.dumps({'prediction': prediction[0][0]})
