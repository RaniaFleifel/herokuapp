# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 03:47:21 2022

@author: Rania Fleifel
"""

from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__,template_folder='templates')
model=pickle.load(open('fish_LinearRegression.model','rb'))

@app.route("/")
def home():
    return render_template('index_new.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)

    output=np.round_(prediction[0],2)
    return render_template('index_new.html',prediction_text='The weight of the fish is {} gms'.format(output[0]))

    #return render_template('indexorig.html')

if __name__=="__main__":
    app.run(debug=True)