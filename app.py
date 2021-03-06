# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['age','anaemia','creatinine','dia','eject','hbp','plates','serum_c','serum_s','sex','smoke','time']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 0:
        res_val = "die"
    else:
        res_val = " not die"

    print(res_val)   

    return render_template('main.html', prediction_text='Patient will {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)
    
