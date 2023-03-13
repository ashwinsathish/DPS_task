import pickle
from urllib import request
import numpy as np

model = pickle.load(open('xgb_pred_model','rb'))
#model = pickle.load(open('nn_model','rb'))

from flask import Flask, render_template, request

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict_accidents():
    category_code = int(request.form.get('category_code'))
    acc_type_code = int(request.form.get('acc_type_code'))
    year = int(request.form.get('year'))
    month = int(request.form.get('month'))

    #prediction
    result = model.predict(np.array([category_code,acc_type_code,year,month]).reshape(1,4))
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)