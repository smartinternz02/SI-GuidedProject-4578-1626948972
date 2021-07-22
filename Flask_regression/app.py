# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:54:19 2021

@author: prads
"""

from flask import Flask , render_template , request

from tensorflow.keras.models import load_model
import joblib
import numpy as np 

model = load_model("regressor.h5")
ct = joblib.load("newcolumn")
#sc = joblib.load("scaler")

app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/login',methods = ["POST","GET"])
def predict():
    if request.method == "POST":
        ms = request.form["ms"]
        ad = request.form["as"]
        rd = request.form["rd"]
        s = request.form["s"]
        data = [[ms,ad,rd,s]]
        print("befor ct", data)
        data = ct.transform(data)
        print("after  ct", data)
        data=  np.asarray(data).astype(np.float32)
        print(data)
        pred = model.predict(data)
        print(pred)
    
    

    return render_template("index.html",value = str(pred[0][0]))
        
if __name__ == "__main__":
    
    app.run(debug = True)


